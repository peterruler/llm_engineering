#!/bin/bash
sudo apt update
sudo apt install -y curl build-essential pkg-config libssl-dev

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

#1
source "$HOME/.cargo/env"

rustc --version
cargo --version
rustup --version

cargo new hello_rust
cd hello_rust
cargo run