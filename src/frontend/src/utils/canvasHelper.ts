const GROUP_COLORS: Record<number, string> = {
  1: '#61dafb', // Root
  2: '#bd34fe', // Category
  3: '#ff4b4b', // Movie
};

interface DrawParams {
  hoverNodeId?: string;
  selectedNodeId?: string;
  highlightNodes: Set<string>;
}

export const drawNodeCanvasObject = (
  node: any,
  ctx: CanvasRenderingContext2D,
  globalScale: number,
  params: DrawParams
) => {
  const { hoverNodeId, selectedNodeId, highlightNodes } = params;
  
  const isHovered = hoverNodeId === node.id;
  const isSelected = selectedNodeId === node.id;
  const isDimmed = (hoverNodeId || selectedNodeId) && !highlightNodes.has(node.id);
  
  const nodeRadius = node.val ?? 6;
  const baseColor = GROUP_COLORS[node.group] || '#999';
  const fillColor = isDimmed ? 'rgba(80, 80, 80, 0.3)' : baseColor;

  // 1. Свечение для активных/наведенных узлов
  if (isHovered || isSelected) {
    ctx.beginPath();
    ctx.arc(node.x, node.y, nodeRadius + (isHovered ? 4 : 3), 0, 2 * Math.PI, false);
    ctx.fillStyle = isSelected ? 'rgba(255, 255, 255, 0.5)' : 'rgba(255, 255, 255, 0.3)';
    ctx.fill();
  }

  // 2. Основная фигура (Квадрат для фильмов, Круг для остальных)
  ctx.beginPath();
  if (node.group === 3) {
    ctx.rect(node.x - nodeRadius, node.y - nodeRadius, nodeRadius * 2, nodeRadius * 2);
  } else {
    ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI, false);
  }
  ctx.fillStyle = fillColor;
  ctx.fill();

  // 3. Белый центр для корневого узла
  if (node.level === 1 && !isDimmed) {
    ctx.beginPath();
    ctx.arc(node.x, node.y, nodeRadius * 0.9, 0, 2 * Math.PI, false);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.fill();
  }

  // 4. Обводка
  ctx.lineWidth = 1.5 / globalScale;
  ctx.strokeStyle = isDimmed ? 'rgba(0,0,0,0)' : '#1a1a1a';
  ctx.stroke();

  // 5. Текст (Имя узла)
  const showText = globalScale > 1.2 || isHovered || isSelected || highlightNodes.has(node.id);
  if (showText && !isDimmed) {
    const label = node.name;
    const fontSize = 12 / globalScale;
    ctx.font = `bold ${fontSize}px Sans-Serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = isHovered || isSelected ? '#ffffff' : 'rgba(255, 255, 255, 0.7)';
    ctx.fillText(label, node.x, node.y + nodeRadius + (10 / globalScale));
  }
};