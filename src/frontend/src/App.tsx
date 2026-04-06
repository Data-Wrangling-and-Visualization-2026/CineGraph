import { useState, useMemo, useEffect } from 'react';
import { Graph } from './components/Graph/Graph';
import { Sidebar } from './components/Sidebar/Sidebar';
import { DetailedPanel } from './components/DetailedPanel/DetailedPanel';
import { fetchGraph } from './api/graph';
import { transformGraphData } from './utils/transform';
import type { MyNode, GraphData } from './types/graph';
import './App.css';

export default function App() {
  const [selectedNode, setSelectedNode] = useState<MyNode | null>(null);
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links:[] });
  
  // Состояние: открыта ли большая панель слева
  const[isDetailsOpen, setIsDetailsOpen] = useState(false);

  useEffect(() => {
    loadGraph(1);
  },[]);

  const loadGraph = async (nodeId: number) => {
    try {
      const data = await fetchGraph(nodeId);
      const transformed = transformGraphData(data);

      setGraphData(prev => {
        const newNodes = [...prev.nodes];
        const newLinks = [...prev.links];

        transformed.nodes.forEach(newNode => {
          if (!newNodes.find(n => n.id === newNode.id)) newNodes.push(newNode);
        });
        transformed.links.forEach(newLink => {
          // Простая проверка, чтобы избежать дубликатов связей
          if (!newLinks.find(l => l.source === newLink.source && l.target === newLink.target)) {
            newLinks.push(newLink);
          }
        });
        return { nodes: newNodes, links: newLinks };
      });
    } catch (e) {
      console.error(e);
    }
  };

  const visibleData = useMemo(() => {
    let visibleNodes: MyNode[] =[];

    if (activeCategory === null) {
      // Показываем корень и его прямых потомков
      visibleNodes = graphData.nodes.filter(node =>
          node.level === 1 || node.level === 2 || node.level === 3
      );
    } else {
      visibleNodes = graphData.nodes.filter(node => {
        if (node.id === activeCategory) return true;

        if (node.level === 2 || node.level === 3) {
          return graphData.links.some(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;

            return (
              (sourceId === activeCategory && targetId === node.id) ||
              (targetId === activeCategory && sourceId === node.id)
            );
          });
        }
        return false;
      });
    }

    const visibleNodeIds = new Set(visibleNodes.map(n => n.id));

    const visibleLinks = graphData.links.filter(link => {
      const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
      const targetId = typeof link.target === 'object' ? link.target.id : link.target;
      return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId);
    });

    return { nodes: visibleNodes, links: visibleLinks };
  }, [graphData, activeCategory]);

  const handleNodeClick = async (node: MyNode) => {
    if (node.id.startsWith('movie-') || node.level === 3) {
      setSelectedNode(node);
      return;
    }

    if (node.level === 1) {
      if (activeCategory === node.id) {
        // Сворачиваем категорию
        setActiveCategory(null);
        setSelectedNode(null);
        setIsDetailsOpen(false);
        return; 
      } else {
        // Разворачиваем категорию
        setActiveCategory(node.id);
        setSelectedNode(null);
        setIsDetailsOpen(false);
      }
    }

    const numericId = Number(node.id);
    if (!isNaN(numericId)) {
      await loadGraph(numericId);
    }
  };

  const handleNodeDoubleClick = (node: MyNode) => {
    if (node.id.startsWith('movie-') || node.level === 3) {
      setSelectedNode(node);
      setIsDetailsOpen(true);
    }
  };

  const handleBackgroundClick = () => {
    setSelectedNode(null);
    setIsDetailsOpen(false);
  };

  return (
    <div className='div-primary'>
      <Graph
        data={visibleData}
        selectedNode={selectedNode}
        isDetailsOpen={isDetailsOpen}
        onNodeClick={handleNodeClick}
        onNodeDoubleClick={handleNodeDoubleClick}
        onBackgroundClick={handleBackgroundClick}
      />

      <Sidebar
        selectedNode={selectedNode}
        isHidden={isDetailsOpen}
        onClose={() => {
          setSelectedNode(null);
          setIsDetailsOpen(false);
        }}
        onOpenDetails={() => {
          setIsDetailsOpen(true);
        }}
      />

      <DetailedPanel 
        node={selectedNode} 
        isOpen={isDetailsOpen} 
        onClose={() => setIsDetailsOpen(false)} 
      />
    </div>
  );
}