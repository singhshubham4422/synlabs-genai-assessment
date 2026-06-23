# Large Language Model (LLM) Fundamentals

Large Language Models (LLMs) have revolutionized natural language processing by leveraging deep neural networks trained on massive text corpora. At the core of all modern LLMs is the Transformer architecture, introduced by Vaswani et al. in 2017. The Transformer replaced recurrent networks (LSTMs) and convolutional layers with self-attention mechanisms, allowing models to process tokens in parallel and capture long-range dependencies across text sequences.

## The Transformer Architecture and Self-Attention

The Transformer architecture is generally divided into an encoder and a decoder. Encoder-only models (like BERT) process inputs bidirectionally and are highly suited for classification and extraction. Decoder-only models (like GPT-3, GPT-4, and Gemini) process inputs autoregressively, predicting the next token in a sequence based only on preceding tokens. Encoder-decoder architectures (like T5) combine both, which is useful for translation and summarization.

Self-attention is the key mathematical mechanism behind the Transformer. Given an input sequence, the model computes three matrices for each token: Queries ($Q$), Keys ($K$), and Values ($V$). The attention score represents how much focus one token should place on another. Mathematically, it is calculated as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $d_k$ is the dimensionality of the key vectors. By scaling the dot product of $Q$ and $K$ by $\sqrt{d_k}$, the model avoids vanishing gradients during training. Multi-Head Attention expands this by running multiple self-attention operations in parallel, allowing the network to capture different semantic and syntactic relationships simultaneously (e.g., matching a verb with its subject while also matching pronouns to their referents).

## Tokenization and Embeddings

Before raw text can be processed by a Transformer, it must be split into numerical representations through a process called tokenization. Modern tokenizers use subword algorithms such as Byte Pair Encoding (BPE), WordPiece, or SentencePiece. These algorithms identify common subwords, prefixes, and suffixes to build a vocabulary of approximately 30,000 to 100,000 unique tokens. This handles out-of-vocabulary words elegantly by breaking unrecognized terms into smaller subword chunks (e.g., "uncompromising" might become "un", "compromise", and "ing").

Once tokenized, each token is mapped to a high-dimensional vector space called an embedding. An embedding is a dense vector representation of a token's semantic meaning. In this geometric space, semantically similar concepts (like "king" and "queen") are positioned close to each other. The dimension of these vectors determines the detail of semantic capturing—ranging from 384 dimensions for lightweight models like `all-MiniLM-L6-v2` up to several thousand dimensions for large frontier models.

## Reinforcement Learning from Human Feedback (RLHF)

Pre-trained base models are skilled at predicting the next word but often write toxic, unhelpful, or hallucinatory content. To align these models with human values, developers use Reinforcement Learning from Human Feedback (RLHF). 

The RLHF process involves three main phases:
1. **Supervised Fine-Tuning (SFT)**: The base model is trained on high-quality demonstration datasets containing prompt-response pairs written by humans.
2. **Reward Model Training**: Human evaluators rate multiple model completions for a single prompt. This preference data trains a Reward Model to output a scalar score representing how helpful or safe a response is.
3. **Reinforcement Learning (RL)**: Using an RL algorithm like Proximal Policy Optimization (PPO), the SFT model is updated using feedback from the Reward Model. The model is penalized if it deviates too far from the initial SFT model (using a KL-divergence penalty) to prevent model collapse or reward hacking. Recently, Direct Preference Optimization (DPO) has emerged as a simplified alternative, optimizing the model directly on human preference pairs without training a separate reward model.
