import ctranslate2
print(f"CTranslate2 version: {ctranslate2.__version__}")
try:
    print("CTranslate2 imported successfully")
except Exception as e:
    print(f"Error: {e}")
