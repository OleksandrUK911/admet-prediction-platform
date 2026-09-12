import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./locales/en.json";
import uk from "./locales/uk.json";

export const LANGUAGE_STORAGE_KEY = "language-preference";
export const SUPPORTED_LANGUAGES = ["en", "uk"];
export const DEFAULT_LANGUAGE = "en";

export function getStoredLanguage() {
  try {
    const value = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    return SUPPORTED_LANGUAGES.includes(value) ? value : null;
  } catch {
    return null;
  }
}

export function setStoredLanguage(language) {
  try {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
  } catch {
    // ignore (e.g. private browsing / storage disabled)
  }
}

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    uk: { translation: uk },
  },
  lng: getStoredLanguage() || DEFAULT_LANGUAGE,
  // Falls back to en whenever a uk key is missing (TODO_i18n_localization.md) -
  // uk translation is secondary/optional and may lag behind new en keys.
  fallbackLng: DEFAULT_LANGUAGE,
  interpolation: { escapeValue: false },
});

export default i18n;
