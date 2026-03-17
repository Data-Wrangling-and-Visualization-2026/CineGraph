import type { MyNode } from '../../types/graph';

interface SidebarProps {
  selectedNode: MyNode | null;
  onClose: () => void;
}

export function Sidebar({ selectedNode, onClose }: SidebarProps) {
  if (!selectedNode) return null;

  return (
    <div
      style={{
        position: 'absolute',
        top: 0,
        right: 0,
        width: '300px',
        background: '#2e2c2c',
        height: '100vh',
        padding: '20px',
        boxShadow: '-4px 0 15px rgba(0,0,0,0.1)'
      }}
    >
      <button onClick={onClose} style={{ marginBottom: '20px' }}>Закрыть</button>
      <h2>{selectedNode.name}</h2>
      <p><b>Группа:</b> {selectedNode.group}</p>
      <p><b>Значимость:</b> {selectedNode.val}</p>
    </div>
  );
}