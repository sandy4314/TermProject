// frontend/src/components/AdminDashboard.js
import React, { useEffect, useState } from 'react';

function AdminDashboard() {
  const [reports, setReports] = useState([]);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(10);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [sentimentFilter, setSentimentFilter] = useState('');
  const [medicineFilter, setMedicineFilter] = useState('');
  const [error, setError] = useState(null);

  const fetchReports = async () => {
    try {
      setLoading(true);
      setError(null);

      const params = new URLSearchParams({
        page: page.toString(),
        page_size: pageSize.toString(),
      });
      if (sentimentFilter) params.append('sentiment', sentimentFilter);
      if (medicineFilter) params.append('medicine', medicineFilter);

      const res = await fetch(`http://localhost:8000/admin/reports?${params.toString()}`);
      if (!res.ok) {
        throw new Error('Failed to fetch reports');
      }
      const data = await res.json();
      setReports(data.items || []);
      setTotal(data.total || 0);
    } catch (err) {
      console.error(err);
      setError('Failed to load reports. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, sentimentFilter, medicineFilter]);

  const totalPages = Math.ceil(total / pageSize) || 1;

  return (
    <div className="border p-6 rounded-lg shadow-lg bg-white mt-6">
      <h2 className="text-2xl font-semibold mb-4">Admin Dashboard – All Reviews</h2>

      {/* Filters */}
      <div className="flex flex-wrap gap-4 mb-4">
        <div>
          <label className="block text-sm font-medium mb-1">Sentiment</label>
          <select
            className="border p-2 rounded"
            value={sentimentFilter}
            onChange={(e) => {
              setPage(1);
              setSentimentFilter(e.target.value);
            }}
          >
            <option value="">All</option>
            <option value="positive">Positive</option>
            <option value="neutral">Neutral</option>
            <option value="negative">Negative</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Medicine</label>
          <input
            className="border p-2 rounded"
            placeholder="Search medicine..."
            value={medicineFilter}
            onChange={(e) => {
              setPage(1);
              setMedicineFilter(e.target.value);
            }}
          />
        </div>
      </div>

      {loading && <p>Loading reports...</p>}
      {error && <p className="text-red-600">{error}</p>}

      {/* Table */}
      {!loading && !error && (
        <>
          <div className="overflow-x-auto">
            <table className="min-w-full border text-sm">
              <thead className="bg-gray-100">
                <tr>
                  <th className="border px-2 py-1">Created</th>
                  <th className="border px-2 py-1">Medicine</th>
                  <th className="border px-2 py-1">Sentiment</th>
                  <th className="border px-2 py-1">Condition</th>
                  <th className="border px-2 py-1">ADRs</th>
                  <th className="border px-2 py-1">Review</th>
                </tr>
              </thead>
              <tbody>
                {reports.length === 0 && (
                  <tr>
                    <td colSpan="6" className="text-center py-3">
                      No reports found.
                    </td>
                  </tr>
                )}
                {reports.map((r) => (
                  <tr key={r.id}>
                    <td className="border px-2 py-1">
                      {r.created_at
                        ? new Date(r.created_at).toLocaleString()
                        : '-'}
                    </td>
                    <td className="border px-2 py-1">{r.medicine || '-'}</td>
                    <td className="border px-2 py-1 capitalize">
                      {r.sentiment}
                    </td>
                    <td className="border px-2 py-1">
                      {r.condition || 'unknown'}
                    </td>
                    <td className="border px-2 py-1">
                      {Array.isArray(r.adrs) ? r.adrs.join(', ') : '-'}
                    </td>
                    <td className="border px-2 py-1 max-w-xs truncate" title={r.review}>
                      {r.review}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between mt-4">
            <p className="text-sm">
              Page {page} of {totalPages} (Total: {total})
            </p>
            <div className="flex gap-2">
              <button
                type="button"
                className="px-3 py-1 border rounded disabled:opacity-50"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                Previous
              </button>
              <button
                type="button"
                className="px-3 py-1 border rounded disabled:opacity-50"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default AdminDashboard;
