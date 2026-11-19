import "@fontsource-variable/inter";
import "@fontsource-variable/space-grotesk";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/layout.css";
import "./styles/components.css";
import "./styles/pages.css";

import { Compass } from "lucide-react";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { EmptyState } from "./components/Feedback";
import { CreatePage } from "./pages/CreatePage";
import { GenerationsPage } from "./pages/GenerationsPage";
import { JobPage } from "./pages/JobPage";
import { SettingsPage } from "./pages/SettingsPage";

function NotFound() {
  return (
    <div className="page">
      <div className="panel">
        <EmptyState icon={Compass} title="Page not found" action={<Link to="/" className="btn btn--primary">Go to Create</Link>} />
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<CreatePage />} />
          <Route path="/generations" element={<GenerationsPage />} />
          <Route path="/generations/:jobId" element={<JobPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  </StrictMode>,
);
