import { useWebSocket } from './hooks/useWebSocket';
import { Header } from './components/layout/Header';
import { Sidebar } from './components/layout/Sidebar';
import { MainContent } from './components/layout/MainContent';
import { DetailPanel } from './components/layout/DetailPanel';

export default function App() {
  // Establish WebSocket connection
  useWebSocket();

  return (
    <div className="dashboard-grid">
      <Header />
      <Sidebar />
      <MainContent />
      <DetailPanel />
    </div>
  );
}
