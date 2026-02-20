import React from 'react';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

function ResultsDisplay({ results }) {
  const sentimentColor = {
    positive: 'text-green-600',
    neutral: 'text-yellow-600',
    negative: 'text-red-600'
  };

  const pieData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        data: [
          results.sentiment === 'positive' ? 1 : 0,
          results.sentiment === 'neutral' ? 1 : 0,
          results.sentiment === 'negative' ? 1 : 0
        ],
        backgroundColor: ['#36A2EB', '#FFCE56', '#FF6384'],
        borderColor: ['#ffffff', '#ffffff', '#ffffff'],
        borderWidth: 1
      }
    ]
  };

  const pieOptions = {
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#333',
          font: {
            size: 14
          }
        }
      },
      tooltip: {
        backgroundColor: '#333',
        titleColor: '#fff',
        bodyColor: '#fff'
      }
    },
    maintainAspectRatio: false
  };

  return (
    <div className="border p-6 rounded-lg shadow-lg bg-white">
      <h2 className="text-2xl font-semibold mb-4">Analysis Results</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <p className="mb-2"><strong>Review:</strong> {results.original_review}</p>
          <p className="mb-2"><strong>Sentiment:</strong> <span className={sentimentColor[results.sentiment]}>{results.sentiment}</span></p>
          <p className="mb-2"><strong>Condition:</strong> {results.condition}</p>
          <p className="mb-2"><strong>ADRs:</strong> {results.adrs.join(', ')}</p>
        </div>
        <div className="h-64">
          <Pie data={pieData} options={pieOptions} />
        </div>
      </div>
    </div>
  );
}

export default ResultsDisplay;
