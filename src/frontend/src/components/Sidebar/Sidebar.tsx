import type { MyNode } from '../../types/graph';
import styles from './Sidebar.module.css';

interface SidebarProps {
  selectedNode: MyNode | null;
  onClose: () => void;
}

export function Sidebar({ selectedNode, onClose }: SidebarProps) {
  if (!selectedNode) return null;

  return (
    <div className={styles.div_sidebar}>
      <button onClick={onClose} style={{ marginBottom: '20px' }}>Закрыть</button>
      <h2>{selectedNode.name}</h2>
      <p><b>Группа:</b> {selectedNode.group}</p>
      <p><b>Значимость:</b> {selectedNode.val}</p>
    </div>
  );
}