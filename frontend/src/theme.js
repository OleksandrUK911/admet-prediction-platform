// Theme preference persistence + application.
// Default (no stored preference) follows the OS via the
// `prefers-color-scheme` media query handled in theme.css.
// Once the user explicitly toggles, we force light/dark via
// `data-theme` on <html>, which theme.css overrides accordingly.

export const THEME_STORAGE_KEY = "theme-preference";

export function getStoredTheme() {
  try {
    const value = localStorage.getItem(THEME_STORAGE_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

export function setStoredTheme(theme) {
  try {
    if (theme === "light" || theme === "dark") {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } else {
      localStorage.removeItem(THEME_STORAGE_KEY);
    }
  } catch {
    // ignore (e.g. private browsing / storage disabled)
  }
}

export function applyTheme(theme) {
  if (theme === "light" || theme === "dark") {
    document.documentElement.dataset.theme = theme;
  } else {
    delete document.documentElement.dataset.theme;
  }
}

// Applies the stored preference (if any) immediately - call this as early
// as possible (before React renders) to avoid a flash of the wrong theme.
export function applyStoredTheme() {
  const stored = getStoredTheme();
  applyTheme(stored);
  return stored;
}
