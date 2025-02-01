const translations: { [key: string]: { [key: string]: string } } = {
    hi: {
      "Welcome to BrightPath": "ब्राइटपाथ में आपका स्वागत है",
      "Empowering young minds through interactive learning": "इंटरैक्टिव लर्निंग के माध्यम से युवा दिमागों को सशक्त बनाना",
      "BrightPath Games": "ब्राइटपाथ गेम्स",
      "Fun and interactive games to boost cognitive skills": "संज्ञानात्मक कौशल बढ़ाने के लिए मजेदार और इंटरैक्टिव गेम",
      "Progress Tracker": "प्रगति ट्रैकर",
      "Monitor your child's learning journey": "अपने बच्चे की सीखने की यात्रा की निगरानी करें",
      "BrightPath Stories": "ब्राइटपाथ कहानियाँ",
      "Engaging tales that teach and entertain": "आकर्षक कहानियाँ जो सिखाती और मनोरंजन करती हैं",
      "Social Skills Builder": "सामाजिक कौशल बिल्डर",
      "Learn to navigate social situations with confidence": "आत्मविश्वास के साथ सामाजिक स्थितियों को नेविगेट करना सीखें",
    },
    kn: {
      "Welcome to BrightPath": "ಬ್ರೈಟ್‌ಪಾಥ್‌ಗೆ ಸುಸ್ವಾಗತ",
      "Empowering young minds through interactive learning": "ಸಂವಾದಾತ್ಮಕ ಕಲಿಕೆಯ ಮೂಲಕ ಯುವ ಮನಸ್ಸುಗಳನ್ನು ಸಬಲೀಕರಣಗೊಳಿಸುವುದು",
      "BrightPath Games": "ಬ್ರೈಟ್‌ಪಾಥ್ ಆಟಗಳು",
      "Fun and interactive games to boost cognitive skills": "ಜ್ಞಾನಾತ್ಮಕ ಕೌಶಲ್ಯಗಳನ್ನು ಹೆಚ್ಚಿಸಲು ಮಜಾ ಮತ್ತು ಸಂವಾದಾತ್ಮಕ ಆಟಗಳು",
      "Progress Tracker": "ಪ್ರಗತಿ ಟ್ರ್ಯಾಕರ್",
      "Monitor your child's learning journey": "ನಿಮ್ಮ ಮಗುವಿನ ಕಲಿಕೆಯ ಪ್ರಯಾಣವನ್ನು ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ",
      "BrightPath Stories": "ಬ್ರೈಟ್‌ಪಾಥ್ ಕಥೆಗಳು",
      "Engaging tales that teach and entertain": "ಕಲಿಸುವ ಮತ್ತು ಮನರಂಜನೆ ನೀಡುವ ಆಕರ್ಷಕ ಕಥೆಗಳು",
      "Social Skills Builder": "ಸಾಮಾಜಿಕ ಕೌಶಲ್ಯ ನಿರ್ಮಾಪಕ",
      "Learn to navigate social situations with confidence": "ಆತ್ಮವಿಶ್ವಾಸದಿಂದ ಸಾಮಾಜಿಕ ಸನ್ನಿವೇಶಗಳನ್ನು ನ್ಯಾವಿಗೇಟ್ ಮಾಡಲು ಕಲಿಯಿರಿ",
    },
    // Add more languages here...
  }
  
  export function translate(text: string, language: string): string {
    if (language === "en") return text
    return translations[language]?.[text] || text
  }
  
  