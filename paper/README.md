# AfriVL-Bench: Grounding Visual Concepts in African Languages

> **Working Paper / Call for Participation**  
> **Contact:** [mich@afrispeech.org](mailto:mich@afrispeech.org)

---

## Abstract

Lexical translation frequently maps nouns and conceptual entities to common cross-lingual equivalents, masking fundamental differences in the physical, perceptual, and cultural referents that speakers hold in mind. For example, during a bilingual conversation where an English translator renders the Akan Twi word *bayerɛ* as *yam*, the conversational partners operate under divergent conceptual models: an English speaker often envisions generic root vegetables or sweet potatoes, whereas a native Twi speaker refers specifically to the West African yam (*Dioscorea*), distinct in form, preparation, and culinary role in Ghana. Over time, conventions established during the historical translation of low-resource African languages into dominant high-resource languages have codified these approximations as equivalence, divorcing vocabulary from local material reality.

This research introduces **AfriVL-Bench**, a benchmark evaluating the ability of multimodal Large Language Models (VLMs / MLLMs) to associate indigenous language terminology directly with the authentic, localized visual referents identified by native speakers. Rather than evaluating translation solely as text-to-text token prediction or matching against Western stock photography, we establish a human-grounded evaluation protocol across nine African languages (Akan Twi, Hausa, Igbo, kiSwahili, Amharic, Wolof, Mandinka, isiXhosa, and isiZulu). By evaluating models on their capacity to distinguish culturally authentic visual representations from generalized visual distractors, this benchmark probes whether state-of-the-art multimodal models possess genuine regional contextual understanding or merely regurgitate high-resource lexical mappings.

---

## Key Research Questions

1. **Perceptual Disconnect in Cross-Lingual Vision Models:** Do vision-language models trained predominantly on Western web data associate African language nouns with culturally accurate local referents, or do they default to Westernized approximations?
2. **Symmetrical Human-VLM Benchmarking:** When given identical visual choices grounded in target countries (Ghana, Nigeria, Kenya, Tanzania, Ethiopia, Senegal, Gambia, South Africa), how closely does model selection align with native-speaker consensus?
3. **Regional and Category Variations:** Across categories (agriculture, cooking, tools, attire, community life), where do current multimodal architectures succeed, and where does lexical translation fail to translate into visual grounding?

---

## Call for Participation & Collaboration

We are actively collaborating with researchers, linguists, native-speaker evaluators, and machine learning practitioners to expand this work into a full research paper.

If you are interested in participating—as a **co-author**, **evaluator**, **dataset contributor**, or **reviewer**—please contact:

📧 **[mich@afrispeech.org](mailto:mich@afrispeech.org)**

Please include:
- Your name, affiliation, and background.
- Language(s) of interest / expertise (e.g., Twi, Hausa, Swahili, Yoruba, Amharic, etc.).
- Area of interest (experimental evaluation, linguistic analysis, dataset curation, or write-up).

---

## Project Resources

- **GitHub Repository:** [AfriSpeech/afrivl-bench](https://github.com/AfriSpeech/afrivl-bench)
- **Human Ground Truth Space:** [AfriSpeech/afrivl-bench-human-eval](https://huggingface.co/spaces/AfriSpeech/afrivl-bench-human-eval)
- **Submissions Dataset:** [AfriSpeech/afrivl-bench-human-eval (Dataset)](https://huggingface.co/datasets/AfriSpeech/afrivl-bench-human-eval)
- **Benchmark Dashboard:** [AfriSpeech/afrivl-bench (Space)](https://huggingface.co/spaces/AfriSpeech/afrivl-bench)
