# Self Attention

## Trade‑offs and Mistakes

- **Major limitations** – Self‑attention scales as O(n²) in sequence length, causing quadratic memory and compute costs; it lacks inherent locality, so long‑range dependencies can dominate short‑range patterns, leading to over‑fitting on small datasets.  
- **Common mistakes** –

## Conclusion

Self‑attention replaces fixed‑size context windows with dynamic, query‑driven weighting, enabling models to capture long‑range dependencies efficiently. It relies on scaled dot‑product attention, multi‑head parallelism, and residual‑layer normalization to maintain gradient flow. The key architectural benefit is linear‑time complexity in sequence length, making transformers scalable to massive inputs.  

**Takeaways:**  
- Attention scores are computed as softmax(QKᵀ/√dₖ), allowing each token to attend to all others.  
- Multi‑head attention splits the representation into sub‑spaces, enriching expressiveness.  
- Layer normalization and residual connections stabilize training.  

**Next steps:** Study positional encodings, relative attention, and efficient transformer variants (e.g., Linformer, Performer) to further reduce memory and compute overhead.







## Trade‑offs and Mistakes

- **Major limitations** – Self‑attention scales as O(n²) in sequence length, causing quadratic memory and compute costs; it lacks inherent locality, so long‑range dependencies can dominate short‑range patterns, leading to over‑fitting on small datasets.  
- **Common mistakes** –









## Trade‑offs and Mistakes

- **Major limitations** – Self‑attention scales as O(n²) in sequence length, causing quadratic memory and compute costs; it lacks inherent locality, so long‑range dependencies can dominate short‑range patterns, leading to over‑fitting on small datasets.  
- **Common mistakes** –

## Conclusion

Self‑attention replaces fixed‑size context windows with dynamic, query‑driven weighting, enabling models to capture long‑range dependencies efficiently. It relies on scaled dot‑product attention, multi‑head parallelism, and residual‑layer normalization to maintain gradient flow. The key architectural benefit is linear‑time complexity in sequence length, making transformers scalable to massive inputs.  

**Takeaways:**  
- Attention scores are computed as softmax(QKᵀ/√dₖ), allowing each token to attend to all others.  
- Multi‑head attention splits the embedding space, enriching representational capacity.  
- Layer normalization and residual connections stabilize training.  

**Next steps:** Study positional encodings, relative attention, and efficient transformer variants (e.g., Linformer, Performer) to further reduce memory and compute overhead.







## Trade‑offs and Mistakes

- **Major limitations** – Self‑attention scales as O(n²) in sequence length, causing quadratic memory and compute costs; it lacks inherent locality, so long‑range dependencies can dominate short‑range patterns, leading to over‑fitting on small datasets.  
- **Common mistakes** –

## Conclusion

Self‑attention replaces fixed‑size context windows with dynamic, query‑driven weighting, enabling models to capture long‑range dependencies efficiently. It relies on scaled dot‑product attention, multi‑head parallelism, and residual‑layer normalization to maintain gradient flow. The key architectural benefit is linear‑time complexity in sequence length, making transformers scalable to massive inputs.  

**Takeaways:**  
- Attention scores are computed as softmax(QKᵀ/√dₖ), allowing each token to attend to all others.  
- Multi‑head attention splits the representation into sub‑spaces, enriching expressiveness.  
- Layer normalization and residual connections stabilize training.  

**Next steps:** Study positional encodings, relative attention, and efficient transformer variants (e.g., Linformer, Performer) to further reduce memory and compute overhead.







## Trade‑offs and Mistakes

- **Major limitations** – Self‑attention scales as O(n²) in sequence length, causing quadratic memory and compute costs; it lacks inherent locality, so long‑range dependencies can dominate short‑range patterns, leading to over‑fitting on small datasets.  
- **Common mistakes** –

## Conclusion

Self‑attention replaces fixed‑size context windows with dynamic, query‑driven weighting, enabling models to capture long‑range dependencies efficiently. It relies on scaled dot‑product attention, multi‑head parallelism, and residual‑layer normalization to maintain gradient flow. The key architectural benefit is linear‑time complexity in sequence length, making transformers scalable to massive inputs.  

**Takeaways:**  
- Attention scores are computed as softmax(QKᵀ/√dₖ), allowing each token to attend to all others.  
- Multi‑head attention splits the embedding space, enriching representational capacity.  
- Layer normalization and residual connections stabilize training.  

**Next steps:** Study positional encodings, relative attention, and efficient transformer variants (e.g., Linformer, Performer) to further reduce memory and compute overhead.
