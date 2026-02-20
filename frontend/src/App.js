import React, { useState, useEffect } from 'react';
import ReviewForm from './components/ReviewForm';
import ResultsDisplay from './components/ResultsDisplay';
import AdminDashboard from './components/AdminDashboard';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [medicines, setMedicines] = useState([]);
  const [showAdmin, setShowAdmin] = useState(false);

  useEffect(() => {
    const fetchMedicines = async () => {
      try {
        const res = await fetch('http://localhost:8000/medicines');
        if (!res.ok) throw new Error('Failed to load medicines');
        const data = await res.json();
        setMedicines(data);
      } catch (err) {
        console.error('Error loading medicines:', err);
      }
    };
    fetchMedicines();
  }, []);

  const handleSubmit = async (review, medicine) => {
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ review, medicine }),
      });
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to analyze review. Please try again.');
    }
  };

  return (
    <div className="container mx-auto p-4">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-3xl font-bold text-center flex-1">
          Drug Review Classifier
        </h1>
        <button
          type="button"
          onClick={() => setShowAdmin((prev) => !prev)}
          className="ml-4 bg-gray-800 text-white px-3 py-2 rounded-lg text-sm"
        >
          {showAdmin ? 'User View' : 'Admin Dashboard'}
        </button>
      </div>

      {!showAdmin && (
        <>
          <ReviewForm onSubmit={handleSubmit} medicines={medicines} />
          {results && <ResultsDisplay results={results} />}
        </>
      )}

      {showAdmin && <AdminDashboard />}
    </div>
  );
}

export default App;
