import { lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import ErrorBoundary from './components/ErrorBoundary';

const BuildSession = lazy(() => import('./pages/BuildSession'));
const GuidancePage = lazy(() => import('./pages/GuidancePage'));
const ExplorePage = lazy(() => import('./pages/ExplorePage'));
const About = lazy(() => import('./pages/About'));

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <Suspense fallback={null}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/build" element={<BuildSession />} />
            <Route path="/guide/:buildId" element={<GuidancePage />} />
            <Route path="/explore" element={<ExplorePage />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </Suspense>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
