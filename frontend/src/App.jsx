import { BrowserRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AppShell } from "./components/AppShell";
import { PredictPage } from "./pages/PredictPage";
import { HistoryPage } from "./pages/HistoryPage";
import { ProfileDetailPage } from "./pages/ProfileDetailPage";
import { AboutPage } from "./pages/AboutPage";

// One QueryClient for the app's lifetime. Its in-memory cache is what
// lets repeat requests (e.g. re-submitting the same SMILES, or
// revisiting History/About) resolve without another network round trip.
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // These endpoints aren't polled/streamed elsewhere, so a query
      // is considered fresh indefinitely once fetched this session;
      // a manual refetch (e.g. History's "retry") still works.
      staleTime: Infinity,
      retry: false,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<PredictPage />} />
            <Route path="history" element={<HistoryPage />} />
            <Route path="profile/:id" element={<ProfileDetailPage />} />
            <Route path="about" element={<AboutPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
