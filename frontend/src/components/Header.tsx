import { Link, useLocation } from 'react-router-dom';

const Header: React.FC = () => {
  const location = useLocation();

  const navLink = (to: string, label: string) => {
    const isActive = location.pathname === to;
    return (
      <li>
        <Link
          to={to}
          className={`relative px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${
            isActive
              ? 'text-white bg-white/10'
              : 'text-gray-400 hover:text-white hover:bg-white/5'
          }`}
        >
          {label}
          {isActive && (
            <span className="absolute -bottom-[13px] left-1/2 -translate-x-1/2 w-6 h-0.5 bg-blue-500 rounded-full" />
          )}
        </Link>
      </li>
    );
  };

  return (
    <header className="glass sticky top-0 z-50 border-b border-white/5">
      <div className="container mx-auto px-4 py-3 flex justify-between items-center">
        <Link to="/" className="flex items-center gap-2 group">
          {/* LEGO brick icon */}
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-lego-red to-lego-orange flex items-center justify-center shadow-lg shadow-lego-red/20 group-hover:shadow-lego-red/40 transition-shadow">
            <div className="grid grid-cols-2 gap-0.5">
              <div className="w-1.5 h-1.5 rounded-full bg-white/80" />
              <div className="w-1.5 h-1.5 rounded-full bg-white/80" />
              <div className="w-1.5 h-1.5 rounded-full bg-white/80" />
              <div className="w-1.5 h-1.5 rounded-full bg-white/80" />
            </div>
          </div>
          <span className="text-xl font-bold text-gradient tracking-tight">
            LegoGen
          </span>
        </Link>

        <nav>
          <ul className="flex items-center gap-1">
            {navLink('/', 'Home')}
            {navLink('/build', 'Build')}
            {navLink('/about', 'About')}
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;
