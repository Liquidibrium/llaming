# from hf_transfer import download
#
# # Download the pretrained model
# download('https://huggingface.co/rustformers/stablelm-ggml/resolve/main/stablelm-base-alpha-3b-q4_0.bin?download=true',
#          'models/stablelm-base-alpha-3b-q4_0.bin',
#          max_files=1, chunk_size=1024, progress=True
#          )

from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="rustformers/open-llama-ggml",
                filename="open_llama_3b-q5_1-ggjt.bin",
                local_dir="models")
