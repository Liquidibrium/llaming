use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use llm::{InferenceFeedback, InferenceResponse, Model, Prompt};
use llm_samplers::prelude::SamplerChain;

// #[derive(Parser)]
// struct Args {
//     model_architecture: llm::ModelArchitecture,
//     model_path: PathBuf,
//     #[arg(long, short = 'v')]
//     pub tokenizer_path: Option<PathBuf>,
//     #[arg(long, short = 'r')]
//     pub tokenizer_repository: Option<String>,
//     #[arg(long, short = 'q')]
//     pub query: Option<String>,
//     #[arg(long, short = 'c')]
//     pub comparands: Vec<String>,
// }

fn main() {
// load a GGML model from disk
    let llama = llm::load::<llm::models::Llama>(
        // path to GGML file
        std::path::Path::new("models/open_llama_3b-q5_1-ggjt.bin"),
        llm::TokenizerSource::Embedded,
        // llm::ModelParameters
        Default::default(),
        // load progress callback
        llm::load_progress_callback_stdout,
    )
        .unwrap_or_else(|err| panic!("Failed to load model: {err}"));

// use the model to generate text from a prompt
    let mut session = llama.start_session(Default::default());
    let res = session.infer::<std::convert::Infallible>(
        // model to use for text generation
        &llama,
        // randomness provider
        &mut rand::thread_rng(),
        // the prompt to use for text generation, as well as other
        // inference parameters
        &llm::InferenceRequest {
            prompt: Prompt::from("Rust is a cool programming language because"),
            play_back_previous_tokens: false,
            maximum_token_count: None,
            parameters: &llm::InferenceParameters {
                sampler: Arc::new(Mutex::new(create_sampler())),
            },
        },
        // llm::OutputRequest
        &mut Default::default(),
        // output callback
        |t| {
            match t {
                InferenceResponse::PromptToken(t) | InferenceResponse::InferredToken(t) => {
                    print!("{t}");
                    std::io::stdout().flush().unwrap();
                    Ok(InferenceFeedback::Continue)
                }
                InferenceResponse::EotToken => {
                    println!("\n\nEnd of text");
                    Ok(InferenceFeedback::Halt)
                }
                InferenceResponse::SnapshotToken(t) => {
                    println!("Other token: {}", t);
                    Ok(InferenceFeedback::Continue)
                }
            }
        },
    );

    match res {
        Ok(result) => println!("\n\nInference stats:\n{result}"),
        Err(err) => println!("\n{err}"),
    }
    println!("Hello, world!");
}

fn create_sampler() -> SamplerChain {
    let mut sc = SamplerChain::new()
        + llm_samplers::samplers::SampleFlatBias::new([(3, f32::NEG_INFINITY)]);
    sc += llm_samplers::samplers::SampleTemperature::new(0.8);
    sc.push_sampler(llm_samplers::samplers::SampleGreedy::new());
    sc
}
