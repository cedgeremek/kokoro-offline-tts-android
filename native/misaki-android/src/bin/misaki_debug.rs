use misaki_android::tokenizer;

fn main() {
    let text = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let tokens = tokenizer::tokenize(&text);
    eprintln!("tokenize: {tokens:#?}");
    let folded = tokenizer::fold_left(tokens);
    eprintln!("fold_left: {folded:#?}");
    let retokenized = tokenizer::retokenize(folded);
    eprintln!("retokenize: {retokenized:#?}");
}
