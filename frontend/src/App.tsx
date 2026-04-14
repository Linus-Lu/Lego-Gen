import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import BuildSession from './pages/BuildSession';
import About from './pages/About';
import ErrorBoundary from './components/ErrorBoundary';

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/build" element={<BuildSession />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
