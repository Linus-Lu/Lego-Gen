import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import BuildSession from './pages/BuildSession';
import GuidancePage from './pages/GuidancePage';
import ExplorePage from './pages/ExplorePage';
import About from './pages/About';
import ErrorBoundary from './components/ErrorBoundary';

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/build" element={<BuildSession />} />
          <Route path="/guide/:buildId" element={<GuidancePage />} />
          <Route path="/explore" element={<ExplorePage />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
