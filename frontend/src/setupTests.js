import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
// Without this, react-i18next's t() returns raw keys ("predict.title")
// instead of resolved English strings in tests - components render fine
// in the real app (main.jsx imports "./i18n"), but nothing does that
// for the jsdom test environment on its own.
import "./i18n";

afterEach(() => {
  cleanup();
});
