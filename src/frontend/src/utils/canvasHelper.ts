// src/utils/canvasHelper.ts

// Расширенная палитра: уникальный цвет для каждой категории
const CATEGORY_PALETTE = [
  '#F0F7FF', // 50  (Фон)
  '#D9EBFF', // 100
  '#BADAFF', // 200
  '#8BBEF8', // 300
  '#65A5F2', // 400
  '#4A90E2', // 500 (Ваш основной цвет)
  '#357ABD', // 600
  '#24619D', // 700
  '#17497D', // 800
  '#0B3058', // 900 (Глубокий текст)
];

// Кэш: id категории → цвет, чтобы цвет не менялся между рендерами
const categoryColorCache = new Map<string, string>();

export function getCategoryColor(nodeId: string): string {
  if (!categoryColorCache.has(nodeId)) {
    // Детерминированный хэш по id
    let hash = 0;
    for (let i = 0; i < nodeId.length; i++) {
      hash = (hash * 31 + nodeId.charCodeAt(i)) >>> 0;
    }
    categoryColorCache.set(nodeId, CATEGORY_PALETTE[hash % CATEGORY_PALETTE.length]);
  }
  return categoryColorCache.get(nodeId)!;
}

// Получить цвет узла (для использования в linkCanvasObject)
export function getNodeColor(node: any): string {
  if (node.group === 1) return '#FFFFFF';
  if (node.group === 2) return getCategoryColor(String(node.id));
  // Для листьев (group 3) — цвет родительской категории, если есть
  if (node.parentColor) return node.parentColor;
  return '#4A90E2';
}

interface DrawParams {
  hoverNodeId?: string;
  selectedNodeId?: string;
}

export const drawNodeCanvasObject = (
  node: any,
  ctx: CanvasRenderingContext2D,
  globalScale: number,
  params: DrawParams
) => {
  if (typeof node.x !== 'number' || typeof node.y !== 'number') return;

  const { hoverNodeId, selectedNodeId } = params;
  const isHovered = hoverNodeId === node.id;
  const isSelected = selectedNodeId === node.id;
  const isActive = isHovered || isSelected;

  // --- РАЗМЕРЫ ---
  let nodeRadius = 3.5;
  if (node.level === 1) nodeRadius = 14;
  else if (node.group === 2) nodeRadius = Math.max(8, Math.min((node.val || 6), 16));

  const baseColor = getNodeColor(node);

  // === АУРА для категорий ===
  if (node.group === 2) {
    const childCount = node.childCount || 1;
    const auraRadius = nodeRadius + 6 + childCount * 1.5;
    const gradient = ctx.createRadialGradient(
      node.x, node.y, nodeRadius * 0.5,
      node.x, node.y, auraRadius
    );
    gradient.addColorStop(0, `${baseColor}22`);
    gradient.addColorStop(1, `${baseColor}00`);
    ctx.beginPath();
    ctx.arc(node.x, node.y, auraRadius, 0, 2 * Math.PI);
    ctx.fillStyle = gradient;
    ctx.fill();
  }

  // === ПУЛЬСИРУЮЩЕЕ КОЛЬЦО для корня ===
  if (node.level === 1) {
    const t = (Date.now() % 2000) / 2000; // 0..1 за 2 сек
    const pulseRadius = nodeRadius + 8 + Math.sin(t * Math.PI * 2) * 5;
    const alpha = 0.15 + Math.sin(t * Math.PI * 2) * 0.1;
    ctx.beginPath();
    ctx.arc(node.x, node.y, pulseRadius, 0, 2 * Math.PI);
    ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
    ctx.lineWidth = 2 / globalScale;
    ctx.stroke();
  }

  // === ОСНОВНОЙ КРУГ ===
  ctx.beginPath();
  ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI);

  if (isActive) {
    // Светлее при наведении/выборе
    ctx.fillStyle = '#FFFFFF';
  } else {
    ctx.fillStyle = baseColor;
  }
  ctx.fill();

  // Обводка под цвет фона (разделитель)
  ctx.lineWidth = 1.5 / globalScale;
  ctx.strokeStyle = '#121212';
  ctx.stroke();

  // === КОЛЬЦО ФОКУСА ===
  if (isActive) {
    ctx.beginPath();
    ctx.arc(node.x, node.y, nodeRadius + 5 / globalScale, 0, 2 * Math.PI);
    ctx.strokeStyle = baseColor;
    ctx.lineWidth = 1.5 / globalScale;
    ctx.stroke();
  }

  // === ТЕКСТ ===
  const isMovie = node.group === 3;
  const showText =
    isActive ||
    (!isMovie && globalScale > 1.2) ||
    globalScale > 3.0;

  if (showText) {
    const label = node.name || '';
    const fontSize = (isMovie ? 9 : 11) / globalScale;
    const fontWeight = isMovie ? 'normal' : '600';
    ctx.font = `${fontWeight} ${fontSize}px "SF Pro Display", system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const textY = node.y + nodeRadius + (8 / globalScale);
    const textWidth = ctx.measureText(label).width;

    // Пилл-фон для категорий и при наведении
    if (!isMovie || isActive) {
      const padX = 4 / globalScale;
      const padY = 2.5 / globalScale;
      const rx = 3 / globalScale;
      const bx = node.x - textWidth / 2 - padX;
      const by = textY - fontSize / 2 - padY;
      const bw = textWidth + padX * 2;
      const bh = fontSize + padY * 2;

      ctx.beginPath();
      ctx.roundRect(bx, by, bw, bh, rx);
      ctx.fillStyle = isActive ? `${baseColor}DD` : '#121212CC';
      ctx.fill();
    }

    // Текст
    ctx.fillStyle = isActive ? '#121212' : (isMovie ? '#8899AA' : baseColor);
    ctx.fillText(label, node.x, textY);
  }
};