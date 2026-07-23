#![no_std]
#![no_main]

use defmt::*;
use embassy_executor::Spawner;
use embassy_futures::join::join;
use embassy_stm32::usb::{Driver, Instance};
use embassy_stm32::{Config, bind_interrupts, peripherals, usb};
use embassy_usb::Builder;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::driver::EndpointError;
use {defmt_rtt as _, panic_probe as _};

bind_interrupts!(struct Irqs {
    OTG_FS => usb::InterruptHandler<peripherals::USB_OTG_FS>;
});

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    info!("Atom01 firmware booting...");
    info!("  target: STM32H743");
    info!("  framework: Embassy async");

    let mut config = Config::default();
    {
        use embassy_stm32::rcc::*;
        config.rcc.hsi = Some(HSIPrescaler::DIV1);
        config.rcc.csi = true;
        config.rcc.hsi48 = Some(Hsi48Config { sync_from_usb: true });
        config.rcc.pll1 = Some(Pll {
            source: PllSource::HSI,
            prediv: PllPreDiv::DIV4,
            mul: PllMul::MUL50,
            fracn: None,
            divp: Some(PllDiv::DIV2),
            divq: None,
            divr: None,
        });
        config.rcc.sys = Sysclk::PLL1_P;
        config.rcc.ahb_pre = AHBPrescaler::DIV2;
        config.rcc.apb1_pre = APBPrescaler::DIV2;
        config.rcc.apb2_pre = APBPrescaler::DIV2;
        config.rcc.apb3_pre = APBPrescaler::DIV2;
        config.rcc.apb4_pre = APBPrescaler::DIV2;
        config.rcc.voltage_scale = VoltageScale::Scale1;
        config.rcc.mux.usbsel = mux::Usbsel::HSI48;
    }
    let p = embassy_stm32::init(config);

    info!("Init complete — bringing up USB-CDC shell");

    let mut ep_out_buffer = [0u8; 256];
    let mut config = embassy_stm32::usb::Config::default();
    config.vbus_detection = false;

    let driver = Driver::new_fs(p.USB_OTG_FS, Irqs, p.PA12, p.PA11, &mut ep_out_buffer, config);

    let mut config = embassy_usb::Config::new(0xc0de, 0xcafe);
    config.manufacturer = Some("RoboParty");
    config.product = Some("Atom01");
    config.serial_number = Some("12345678");

    let mut config_descriptor = [0; 256];
    let mut bos_descriptor = [0; 256];
    let mut control_buf = [0; 64];

    let mut state = State::new();

    let mut builder = Builder::new(
        driver,
        config,
        &mut config_descriptor,
        &mut bos_descriptor,
        &mut [],
        &mut control_buf,
    );

    let mut class = CdcAcmClass::new(&mut builder, &mut state, 64);
    let mut usb = builder.build();

    let usb_fut = usb.run();

    let echo_fut = async {
        loop {
            class.wait_connection().await;
            info!("USB-CDC shell connected");
            // Send the initial prompt so the user knows the shell is ready.
            // (read_packet would otherwise wait silently for the first byte.)
            // If the host just disconnected, this write fails with Disabled —
            // echo() will also fail immediately and we re-loop.
            let _ = write_best_effort(&mut class, b"$atom01: ").await;
            // echo() only returns Err when the USB endpoint is disabled
            // (host unplugged); BufferOverflow is handled inline.
            if echo(&mut class).await.is_err() {
                info!("USB-CDC shell disconnected");
            }
        }
    };

    join(usb_fut, echo_fut).await;
}

/// Best-effort USB write that survives BufferOverflow without aborting the
/// shell loop. Disabled still propagates (real disconnect).
async fn write_best_effort<'d, T: Instance + 'd>(
    class: &mut CdcAcmClass<'d, Driver<'d, T>>,
    data: &[u8],
) -> Result<(), EndpointError> {
    match class.write_packet(data).await {
        Ok(()) => Ok(()),
        Err(EndpointError::BufferOverflow) => {
            warn!("USB CDC TX buffer overflow — dropped {} bytes", data.len());
            Ok(())
        }
        Err(EndpointError::Disabled) => Err(EndpointError::Disabled),
    }
}

async fn echo<'d, T: Instance + 'd>(
    class: &mut CdcAcmClass<'d, Driver<'d, T>>
) -> Result<(), EndpointError> {
    let mut rx_buf: [u8; 512] = [0; 512];
    let mut line_buf: [u8; 512] = [0; 512];
    let mut line_len: usize = 0;
    let mut last_was_cr: bool = false;
    loop {
        let n = match class.read_packet(&mut rx_buf).await {
            Ok(n) => n,
            Err(EndpointError::BufferOverflow) => {
                warn!("USB CDC RX buffer overflow — dropped packet");
                continue;
            }
            Err(EndpointError::Disabled) => return Err(EndpointError::Disabled),
        };

        for &b in &rx_buf[..n] {
            match b {
                b'\r' | b'\n' => {
                    // Skip the second byte of a CR+LF pair so we don't emit
                    // an empty extra line + duplicate prompt.
                    let is_pair_continuation = b == b'\n' && last_was_cr;
                    last_was_cr = b == b'\r';
                    if is_pair_continuation { continue; }

                    write_best_effort(class, b"\r\n").await?;
                    if line_len > 0 {
                        let line = &line_buf[..line_len];
                        write_best_effort(class, line).await?;
                        line_len = 0;
                    }
                    write_best_effort(class, b"$atom01: ").await?;
                }
                b'\x08' | b'\x7f' => {
                    last_was_cr = false;
                    if line_len > 0 {
                        line_len -= 1;
                        write_best_effort(class, b"\x08 \x08").await?;
                    }
                }
                _ => {
                    last_was_cr = false;
                    if line_len < line_buf.len() {
                        line_buf[line_len] = b;
                        line_len += 1;
                        write_best_effort(class, &[b]).await?;
                    } else {
                        write_best_effort(class, b"\x07").await?;
                    }
                }
            }
        }
    }
}
