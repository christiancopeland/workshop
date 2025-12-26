import os
import openwakeword as oww

# Base package directory
pkg_dir = os.path.dirname(oww.__file__)
print("Package dir:", pkg_dir)

# Models live under resources/models
models_path = os.path.join(pkg_dir, "resources", "models")
print("Models path:", models_path)

print("Files in models dir:")
for f in os.listdir(models_path):
    print(" -", f)
