#!/usr/bin/env python3
"""
Configuration file for Custom Dictionary STT
Edit this file to customize your dictionary terms
"""

# KEYWORDS: Boost recognition of specific terms
# Format: "term:boost_value" (boost_value typically 1.0-3.0)
# Higher boost values increase recognition likelihood
KEYWORDS = [
    # Technology Terms
    "Opus:2.0",
    "Sonnet:2.0",
    "Haiku:2.0",
    "Janus:2.0",
    "Repligate:2.0",
    "Antra:2.0",
    "Aster:2.0",
    "Tel0s:2.0",
    "Lari:2.0",
    "Dario:2.0",
    "Laria:2.0",
    "BorgCord:2.0",
    "Cyborgism:2.0",
    "Borg:2.0",
    "Borgs:2.0",
    "Connectome:2.0",
    "GemPro:2.0",
    "Claude:1.8",
    "Anthropic:1.8",
    "grug:1.5",
    "gorm:1.5",
    "qualia:1.5",

    
    # Company Names (add your company/clients here)
    # "YourCompanyName:2.0",
    # "ClientName:2.0",
    
    # Industry-Specific Terms
    # Medical: "HIPAA:2.0", "EMR:2.0", "EHR:2.0"
    # Legal: "GDPR:2.0", "compliance:1.8", "litigation:1.8"
    # Finance: "fintech:1.8", "blockchain:1.8", "cryptocurrency:1.8"
    
    # Add your custom terms here:
    # "YourTerm:1.5",
]

# REPLACEMENTS: Replace recognized words with preferred versions
# Format: {"incorrect_word": "correct_word"}
# These fix common misrecognition patterns
REPLACEMENTS = {
    # Common tech term fixes
    "deep gram": "Deepgram",
    "deep grand": "Deepgram",
    "pie torch": "PyTorch",
    "tensor flow": "TensorFlow",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "git hub": "GitHub",
    "java script": "JavaScript",
    "type script": "TypeScript",
    "react js": "React",
    "node js": "Node.js",
    "vs code": "VS Code",
    "open ai": "OpenAI",
    "chat gpt": "ChatGPT",
    "gpt 4": "GPT-4",
    "anthropic": "Anthropic",
    
    # Business term fixes
    "see our em": "CRM",
    "sass": "SaaS",
    "b two b": "B2B",
    "b two c": "B2C",
    "are oh i": "ROI",
    "kay p i": "KPI",
    "em v p": "MVP",
    "you i": "UI",
    "you x": "UX",
    
    # Company name fixes (customize these)
    # "your company": "YourCompanyName",
    # "client name": "ClientName",
    
    # Industry-specific fixes
    # Medical
    "hipaa": "HIPAA",
    "hippo": "HIPAA",
    "e m r": "EMR",
    "e h r": "EHR",
    
    # Legal
    "gdpr": "GDPR",
    "gd pr": "GDPR",
    
    # Finance
    "fin tech": "fintech",
    "block chain": "blockchain",
    "crypto currency": "cryptocurrency",
    "bit coin": "Bitcoin",
    
    # Add your custom replacements here:
    # "incorrect_phrase": "CorrectPhrase",
}

# SEARCH TERMS: Highlight specific terms in output
# These terms will be highlighted when they appear in transcripts
SEARCH_TERMS = [
    "API",
    "Deepgram",
    "Python",
    "machine learning",
    "AI",
    "neural network",
    "JavaScript",
    "React",
    "Docker",
    "Kubernetes",
    "AWS",
    "Azure",
    "OpenAI",
    "ChatGPT",
    "GPT",
    
    # Add terms you want to highlight:
    # "YourCompanyName",
    # "ImportantTerm",
]

# INDUSTRY PRESETS
# Uncomment and customize based on your industry

# MEDICAL_KEYWORDS = [
#     "HIPAA:2.0",
#     "EMR:2.0",
#     "EHR:2.0",
#     "patient:1.5",
#     "diagnosis:1.8",
#     "treatment:1.8",
#     "medication:1.8",
#     "prescription:1.8",
#     "healthcare:1.8",
# ]

# LEGAL_KEYWORDS = [
#     "GDPR:2.0",
#     "compliance:1.8",
#     "litigation:1.8",
#     "contract:1.8",
#     "agreement:1.8",
#     "jurisdiction:1.8",
#     "plaintiff:1.8",
#     "defendant:1.8",
# ]

# FINANCE_KEYWORDS = [
#     "fintech:1.8",
#     "blockchain:1.8",
#     "cryptocurrency:1.8",
#     "Bitcoin:1.8",
#     "Ethereum:1.8",
#     "investment:1.5",
#     "portfolio:1.5",
#     "revenue:1.5",
#     "profit:1.5",
# ]

# EDUCATION_KEYWORDS = [
#     "curriculum:1.8",
#     "pedagogy:1.8",
#     "assessment:1.8",
#     "learning:1.5",
#     "student:1.5",
#     "teacher:1.5",
#     "classroom:1.5",
# ]

# To use industry presets, uncomment and add them to KEYWORDS:
# KEYWORDS.extend(MEDICAL_KEYWORDS)
# KEYWORDS.extend(LEGAL_KEYWORDS)
# etc. 