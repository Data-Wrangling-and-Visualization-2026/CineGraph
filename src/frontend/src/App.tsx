import { useState, useMemo } from 'react';
import { Graph } from './components/Graph/Graph';
import { Sidebar } from './components/Sidebar/Sidebar';
import { fullGraphData } from './data/mockData';
import type { MyNode } from './types/graph';

export default function App() {
  const [selectedNode, setSelectedNode] = useState<MyNode | null>(null); // Для сайдбара

  // НОВОЕ СОСТОЯНИЕ: ID узла 1-го уровня, в который мы "провалились"
  const [activeCategory, setActiveCategory] = useState<string | null>(null);

  // ВЫЧИСЛЯЕМ ВИДИМЫЙ ГРАФ
  const visibleData = useMemo(() => {
    let visibleNodes: MyNode[] =[];

    if (activeCategory === null) {
      // 1. ГЛАВНЫЙ ЭКРАН: Показываем ТОЛЬКО узлы 1-го уровня
      visibleNodes = fullGraphData.nodes.filter(node => node.level === 1);
    } else {
      // 2. ЭКРАН КАТЕГОРИИ: Показываем саму категорию и её узлы 2-го уровня
      visibleNodes = fullGraphData.nodes.filter(node => {
        // Оставляем саму нажатую категорию (чтобы она была в центре как "ядро")
        if (node.id === activeCategory) return true;

        // Для узлов 2-го уровня проверяем, связаны ли они с активной категорией
        if (node.level === 2) {
          return fullGraphData.links.some(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;

            return (sourceId === activeCategory && targetId === node.id) ||
                   (targetId === activeCategory && sourceId === node.id);
          });
        }

        // Все остальные узлы (другие категории и чужие дочерние узлы) отбрасываем
        return false;
      });
    }

    const visibleNodeIds = new Set(visibleNodes.map(n => n.id));

    // Оставляем только те связи, где оба конца линии сейчас видимы на экране
    const visibleLinks = fullGraphData.links.filter(link => {
      const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
      const targetId = typeof link.target === 'object' ? link.target.id : link.target;
      return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId);
    });

    return { nodes: visibleNodes, links: visibleLinks };
  },[activeCategory]);

  // ОБРАБОТКА КЛИКОВ
  const handleNodeClick = (node: MyNode) => {
    if (node.level === 1) {
      // Если кликнули на категорию
      if (activeCategory === node.id) {
        // Если она уже активна - выходим на главный экран
        setActiveCategory(null);
        setSelectedNode(null); // Закрываем сайдбар, если был открыт
      } else {
        // Проваливаемся в новую категорию
        setActiveCategory(node.id);
        setSelectedNode(null); // Сбрасываем выбранный узел при смене экрана
      }
    } else {
      // Если кликнули на конечный узел - открываем сайдбар
      setSelectedNode(node);
    }
  };

  const handleBackgroundClick = () => {
    // Клик по фону просто снимает выделение (закрывает сайдбар)
    setSelectedNode(null);
  };

  const handleGoBack = () => {
    setActiveCategory(null);
    setSelectedNode(null);
  };

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', background: '#0f0f11' }}>

      {/* КНОПКА ВОЗВРАТА (появляется только если мы внутри категории) */}
      {activeCategory && (
        <button
          onClick={handleGoBack}
          style={{
            position: 'absolute',
            top: '20px',
            left: '20px',
            zIndex: 10,
            padding: '10px 20px',
            fontSize: '16px',
            background: 'rgba(255, 255, 255, 0.1)',
            color: '#fff',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '8px',
            cursor: 'pointer',
            backdropFilter: 'blur(5px)',
            transition: 'background 0.2s'
          }}
          onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)'}
          onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)'}
        >
          ← Назад к категориям
        </button>
      )}

      <Graph
        data={visibleData}
        selectedNode={selectedNode}
        onNodeClick={handleNodeClick}
        onBackgroundClick={handleBackgroundClick}
      />

      <Sidebar
        selectedNode={selectedNode}
        onClose={() => setSelectedNode(null)}
      />
    </div>
  );
}