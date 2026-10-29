import sys
try:
    import ctranslate2
    from ctranslate2.converters import TransformersConverter
    print("CTranslate2 Converter available")
except ImportError as e:
    print(f"CTranslate2 Converter NOT available: {e}")

try:
    import transformers
    print("Transformers available")
except ImportError as e:
    print(f"Transformers NOT available: {e}")
