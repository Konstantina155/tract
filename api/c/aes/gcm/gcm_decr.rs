use aes_gcm::{
    aead::{KeyInit, generic_array::GenericArray, generic_array::typenum::U16, Error as AeadError},
    Aes256Gcm,
    AeadInPlace,
};
use std::env;
use std::fs::File;
use std::io::Read;

fn decrypt(key: &[u8], iv: &[u8], cipher_text: &mut [u8], additional_data: &[u8], tag: &GenericArray<u8, U16>) -> Result<(), AeadError> {
    let key = GenericArray::from_slice(key);
    let cipher = Aes256Gcm::new(key);

    let nonce = GenericArray::from_slice(iv);

    // Decrypt the cipher_text and verify the authentication tag
    Ok(cipher.decrypt_in_place_detached(nonce, additional_data, cipher_text, tag)?)
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 6 {
        println!("Usage: {} <key> <iv> <aad> <tag> <cipher_text>", args[0]);
        return;
    }

    let mut file2 = File::open(&args[1]).expect("File not found!");
    let mut key = Vec::new();
    file2.read_to_end(&mut key).expect("Error in reading file!");

    let mut file3 = File::open(&args[2]).expect("File not found!");
    let mut iv = Vec::new();
    file3.read_to_end(&mut iv).expect("Error in reading file!");

    let mut file4 = File::open(&args[3]).expect("File not found!");
    let mut additional_data = Vec::new();
    file4.read_to_end(&mut additional_data).expect("Error in reading file!");
    
    println!("Key: {:?}", key);
    println!("IV: {:?}", iv);
    println!("Additional Data: {:?}", additional_data);

    // testing with different buffers to corrupt GCM
    // key = Vec::from(&additional_data[0..32]);
    // iv = Vec::from(&additional_data[0..12]);
    // additional_data = Vec::from([key.clone(), key.clone()].concat());
    // let tag = GenericArray::default(); to test different tag

    for i in (0..(args.len() - 4)/2 +1).step_by(2) {
        println!("Decrypting files: {}, {}", args[i+4], args[i+5]);
        let mut file = File::open(&args[i+5]).expect("File not found!");
        let mut cipher_text = Vec::new();
        file.read_to_end(&mut cipher_text).expect("Error in reading file!");

        let mut file5 = File::open(&args[i+4]).expect("File not found!");
        let mut tag_vec = Vec::new();
        file5.read_to_end(&mut tag_vec).expect("Error in reading file!");
        println!("Tag: {:?}", tag_vec);

        let tag = GenericArray::clone_from_slice(&tag_vec);
        match decrypt(&key, &iv, &mut cipher_text, &additional_data, &tag) {
            Ok(_) => {
                println!("Decryption is correct!");
            },
            Err(_) => {
                println!("Error in decryption process!");
            }
        }
    }
}
