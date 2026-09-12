import { Component } from "react";
import { withTranslation } from "react-i18next";

// Error boundaries must be class components - React has no hook equivalent
// for getDerivedStateFromError/componentDidCatch (still true as of React 19).
// Wrapped around routed page content only (see App.jsx/AppShell.jsx) so the
// disclaimer banner, header nav and language/theme controls stay visible and
// usable even if a page crashes - see frontend-spec/app-shell.md.
class ErrorBoundaryImpl extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // No backend error-reporting endpoint exists; log locally at minimum.
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    const { t } = this.props;
    if (this.state.hasError) {
      return (
        <div
          role="alert"
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 16,
            textAlign: "center",
            padding: "48px 16px",
            color: "var(--text)",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-card)",
          }}
        >
          <h2 style={{ margin: 0 }}>{t("errors.boundaryTitle")}</h2>
          <button type="button" className="icon-button" onClick={this.handleReload}>
            {t("errors.reloadButton")}
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export const ErrorBoundary = withTranslation()(ErrorBoundaryImpl);
