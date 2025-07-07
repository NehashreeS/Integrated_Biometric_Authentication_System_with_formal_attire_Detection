# Integrated Biometric Authentication System with Formal Attire Detection

This project combines biometric authentication using fingerprint recognition with attire validation to ensure users are formally dressed. The system integrates a GUI for user interaction, fingerprint feature extraction and matching, and attire detection from submitted images.

---

## 📁 Project Structure

```
.
├── fingerprint_scans/                # Folder to store registered fingerprint images
├── submitted_images/                 # Folder to store user-submitted images for attire validation
├── chatbot.png                       # Optional chatbot interface image (UI enhancement)
├── README.md                         # This documentation file
├── finger.py                         # Handles fingerprint matching logic
├── fingerprint_feature_extraction.py# Extracts unique features from fingerprint images
├── ii.py                             # Image inference or attire classification (likely)
├── ui.py                             # GUI interface for user interaction
```

---

## 🚀 Features

* **Fingerprint Authentication**: Uses biometric fingerprint scans for secure identification.
* **Attire Verification**: Checks whether the user is in formal attire using image-based analysis.
* **Graphical User Interface**: Streamlined interaction for registration, login, and validation via `ui.py`.
* **Real-time Feedback**: Immediate validation of fingerprint and attire compliance.

---

## 💠 How to Run

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:

   ```bash
   python ui.py
   ```

3. **Register a User**:

   * Upload a fingerprint image to `fingerprint_scans/`
   * Upload a formal attire image to `submitted_images/`

4. **Login Process**:

   * Select fingerprint and image to authenticate.

---

## 🧠 Core Logic

| File                                | Description                                                   |
| ----------------------------------- | ------------------------------------------------------------- |
| `finger.py`                         | Matches input fingerprint against registered ones.            |
| `fingerprint_feature_extraction.py` | Extracts minutiae and unique features from the fingerprint.   |
| `ii.py`                             | Likely handles attire image classification (formal/informal). |
| `ui.py`                             | Launches the user interface.                                  |

---

## ✅ Requirements

* Python 3.7+
* OpenCV
* NumPy
* Scikit-learn or any custom ML model (for attire detection)
* Tkinter or PyQt (for GUI)

---

## 👤 Author

Nehashree S

---

## 📄 License

This project is licensed under the MIT License.

---

## 🔍 Future Work

* Add facial recognition
* Improve attire classification accuracy using deep learning
* Add database integration for scalable deployments
