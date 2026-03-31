import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import BuildSession from './pages/BuildSession';
import About from './pages/About';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/build" element={<BuildSession />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </Router>
  );
}

export default App;
