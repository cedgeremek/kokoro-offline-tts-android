use misaki_android::{english_to_phonemes_for_accent, espeak};
use std::io::{self, BufRead};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: misaki-parity <espeak-library> <espeak-data> <us|gb> [text]");
        std::process::exit(2);
    }
    if let Err(error) = espeak::initialize(&args[1], &args[2]) {
        eprintln!("{error}");
        std::process::exit(3);
    }
    let british = args[3].eq_ignore_ascii_case("gb");
    if args.len() > 4 {
        let text = args[4..].join(" ");
        match english_to_phonemes_for_accent(&text, british) {
            Ok(phonemes) => println!("{phonemes}"),
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(4);
            }
        }
    } else {
        for line in io::stdin().lock().lines() {
            let text = line.expect("read parity input");
            match english_to_phonemes_for_accent(&text, british) {
                Ok(phonemes) => println!("{phonemes}"),
                Err(error) => {
                    eprintln!("{error}");
                    std::process::exit(4);
                }
            }
        }
    }
}
