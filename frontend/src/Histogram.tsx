import React, { useEffect, useRef } from 'react';

interface HistogramProps {
  data: number[];
}

export const Histogram: React.FC<HistogramProps> = ({ data }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    if (!canvasRef.current || !data) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
  
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    
    const maxValue = Math.max(...data);
    
   
    const barWidth = Math.max(1, Math.floor(canvas.width / data.length));
    const barSpacing = 0;
    const scaleFactor = maxValue > 0 ? canvas.height / maxValue : 0;
    
    
    ctx.fillStyle = '#3498db';
    
    for (let i = 0; i < data.length; i++) {
      const barHeight = data[i] * scaleFactor;
      const x = i * (barWidth + barSpacing);
      const y = canvas.height - barHeight;
      
      ctx.fillRect(x, y, barWidth, barHeight);
    }
    
    
    ctx.strokeStyle = '#7f8c8d';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height - 1);
    ctx.lineTo(canvas.width, canvas.height - 1);
    ctx.stroke();
    
   
    ctx.fillStyle = '#2c3e50';
    ctx.font = '10px Arial';
    ctx.fillText('0', 0, canvas.height - 5);
    ctx.fillText('255', canvas.width - 20, canvas.height - 5);
    
    if (maxValue > 0) {
      ctx.fillText(`${maxValue}`, 0, 10);
    }
  }, [data]);
  
  return (
    <canvas 
      ref={canvasRef} 
      className="histogram-canvas" 
      width={400} 
      height={200}
    />
  );
}; 