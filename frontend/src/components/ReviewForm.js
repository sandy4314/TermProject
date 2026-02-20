import React, { useState } from 'react';

function ReviewForm({ onSubmit, medicines }) {
  const [review, setReview] = useState('');
  const [medicine, setMedicine] = useState('');

    const handleSubmit = (e) => {
    e.preventDefault();
    if (review.trim()) {
      onSubmit(review, medicine);
      setReview('');
      // optionally keep medicine
      // setMedicine('');
    } else {
      alert('Please enter a review.');
    }
  };

    return (
    <div className="mb-6">
      <form onSubmit={handleSubmit}>
        <div className="mb-3">
          <label className="block mb-1 font-semibold">Medicine (optional)</label>
          <input
            className="w-full p-2 border rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            list="medicine-options"
            placeholder="Type medicine name..."
            value={medicine}
            onChange={(e) => setMedicine(e.target.value)}
          />
          <datalist id="medicine-options">
            {medicines && medicines.map((name) => (
              <option key={name} value={name} />
            ))}
          </datalist>
        </div>

        <textarea
          className="w-full p-3 border rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows="5"
          placeholder="Enter your drug review here..."
          value={review}
          onChange={(e) => setReview(e.target.value)}
        ></textarea>

        <button
          type="submit"
          className="mt-2 bg-blue-500 text-white p-3 rounded-lg hover:bg-blue-600 transition duration-200"
        >
          Analyze Review
        </button>
      </form>
    </div>
  );
}

export default ReviewForm;
