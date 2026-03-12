import { useRef, useState, useMemo, useCallback, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { ForceGraphMethods } from 'react-force-graph-2d';
import type { MyNode, MyLink, GraphData } from '../../types/graph';

interface GraphProps {
  data: GraphData;
  selectedNode: MyNode | null;
  onNodeClick: (node: MyNode) => void;
  onBackgroundClick: () => void;
}

const GROUP_COLORS: Record<number, string> = {
  1: '#61dafb',
  2: '#bd34fe',
  3: '#ff4b4b',
};

export function Graph({ data, selectedNode, onNodeClick, onBackgroundClick }: GraphProps) {
  const fgRef = useRef<ForceGraphMethods<any, any> | undefined>(undefined);
  
  // Состояние: на какой узел сейчас наведена мышь
  const[hoverNode, setHoverNode] = useState<MyNode | null>(null);

  // Вычисляем, какие узлы и грани нужно подсветить прямо сейчас
  // useMemo нужен, чтобы не пересчитывать это 60 раз в секунду
  const { highlightNodes, highlightLinks } = useMemo(() => {
    const nodes = new Set<string>();
    const links = new Set<any>();

    // Решаем, от какого узла плясать (приоритет у наведения, затем у клика)
    const activeNode = hoverNode || selectedNode;

    if (activeNode) {
      nodes.add(activeNode.id);
      
      // Ищем все связи этого узла
      data.links.forEach((link: any) => {
        // Библиотека мутирует links, заменяя ID на объекты узлов, поэтому делаем проверку:
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        const targetId = typeof link.target === 'object' ? link.target.id : link.target;

        if (sourceId === activeNode.id || targetId === activeNode.id) {
          links.add(link);
          nodes.add(sourceId);
          nodes.add(targetId);
        }
      });
    }

    return { highlightNodes: nodes, highlightLinks: links };
  },[data, hoverNode, selectedNode]);

  // Меняем курсор на "пальчик" при наведении на узел
  useEffect(() => {
    document.body.style.cursor = hoverNode ? 'pointer' : 'default';
  }, [hoverNode]);

  return (
    <ForceGraph2D
      ref={fgRef}
      graphData={data}
      
      // --- НАСТРОЙКА ГРАНЕЙ (ЛИНИЙ) ---
      // Цвет линии: если подсвечена - белая непрозрачная, иначе - серая полупрозрачная
      linkColor={(link: any) => highlightLinks.has(link) ? 'rgba(255, 255, 255, 0.8)' : 'rgba(255, 255, 255, 0.1)'}
      // Толщина линии
      linkWidth={(link: any) => highlightLinks.has(link) ? 3 : 1}
      // Добавим частицы (бегущие точки по линиям) для выделенных связей!
      linkDirectionalParticles={(link: any) => highlightLinks.has(link) ? 4 : 0}
      linkDirectionalParticleWidth={3}
      linkDirectionalParticleSpeed={0.01}

      // --- ОБРАБОТЧИКИ СОБЫТИЙ ---
      onNodeClick={(node) => onNodeClick(node as MyNode)}
      onBackgroundClick={onBackgroundClick}
      // Когда мышь заходит на узел / уходит с него
      onNodeHover={(node) => setHoverNode((node as MyNode) || null)}

      // --- КАСТОМНАЯ ОТРИСОВКА УЗЛОВ ---
      nodeCanvasObject={(node: any, ctx, globalScale) => {
        const isHovered = hoverNode?.id === node.id;
        const isSelected = selectedNode?.id === node.id;
        
        // Логика затемнения: если есть активный узел, но текущий узел не в соседях - затемняем его
        const isDimmed = (hoverNode || selectedNode) && !highlightNodes.has(node.id);

        const nodeRadius = (node.size == null ? 5 : node.size); 
        const baseColor = GROUP_COLORS[node.group] || '#999';

        // 1. Отрисовка свечения (Halo) вокруг выделенного/наведенного узла
        if (isHovered || isSelected) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, nodeRadius + (isHovered ? 4 : 3), 0, 2 * Math.PI, false);
          ctx.fillStyle = isSelected ? 'rgba(255, 255, 255, 0.5)' : 'rgba(255, 255, 255, 0.3)';
          ctx.fill();
        }

        // 2. Отрисовка самого кружочка узла
        ctx.beginPath();
        ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI, false);
        // Если узел затемнен, делаем его прозрачным, иначе берем его цвет
        ctx.fillStyle = isDimmed ? 'rgba(80, 80, 80, 0.3)' : baseColor;
        ctx.fill();

        if (node.level === 1 && !isDimmed) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, nodeRadius * 0.9, 0, 2 * Math.PI, false);
          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.fill();
        }

        // 3. Отрисовка обводки (бордера) кружочка
        ctx.lineWidth = 1.5 / globalScale; // Толщина обводки не меняется при зуме
        ctx.strokeStyle = isDimmed ? 'rgba(0,0,0,0)' : '#1a1a1a';
        ctx.stroke();

        // 4. Отрисовка текста (имени)
        // Текст показываем только если мы близко (globalScale > 1.2) ИЛИ узел подсвечен
        const showText = globalScale > 1.2 || isHovered || isSelected || highlightNodes.has(node.id);
        
        if (showText && !isDimmed) {
          const label = node.name;
          const fontSize = 12 / globalScale;
          ctx.font = `bold ${fontSize}px Sans-Serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          
          // Белый текст для выделенных, серый для остальных
          ctx.fillStyle = isHovered || isSelected ? '#ffffff' : 'rgba(255, 255, 255, 0.7)';
          ctx.fillText(label, node.x, node.y + nodeRadius + (10 / globalScale));
        }
      }}

      // Цвет фона космоса
      backgroundColor="#0f0f11"
    />
  );
}