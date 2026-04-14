import React, { useState, useRef } from 'react';
import styles from './Modals.module.css';
import { addMovie } from '../../api/graph';

interface AddMovieModalProps {
  onClose: () => void;
}

export function AddMovieModal({ onClose }: AddMovieModalProps) {
  const [title, setTitle] = useState('');
  const [subtitles, setSubtitles] = useState('');
  const [year, setYear] = useState('');
  const [releaseDate, setReleaseDate] = useState('');
  const [runtime, setRuntime] = useState('');
  const [budget, setBudget] = useState('');
  const [overview, setOverview] = useState('');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      const text = evt.target?.result;
      if (typeof text === 'string') {
        setSubtitles(text);
      }
    };
    reader.onerror = () => {
      setError("Error reading file");
    };
    
    reader.readAsText(file);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !subtitles.trim()) return;

    setLoading(true);
    setError(null);

    const payload = {
      title: title.trim(),
      year: year ? parseInt(year) : new Date().getFullYear(),
      subtitles: subtitles.trim(),
      other_data: {
        budget: budget ? parseInt(budget) : undefined,
        overview: overview.trim() || undefined,
        release_date: releaseDate || undefined,
        runtime: runtime ? parseInt(runtime) : undefined,
        status: "Released"
      }
    };

    try {
      await addMovie(payload);
      setSuccess(true);
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (err: any) {
      setError(err.message || "An error occurred during addition");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div 
        className={styles.modal} 
        style={{ maxHeight: '95vh', overflowY: 'auto' }} 
        onClick={(e) => e.stopPropagation()}
      >
        <button className={styles.close_btn} onClick={onClose}>×</button>
        <h2 className={styles.title}>Add Movie</h2>

        <form onSubmit={handleSubmit}>
          <div className={styles.form_group}>
            <label>Movie Title *</label>
            <input 
              required 
              type="text" 
              className={styles.input} 
              value={title} 
              onChange={e => setTitle(e.target.value)} 
              placeholder="e.g. Jaws"
            />
          </div>

          <div style={{ display: 'flex', gap: '15px' }}>
            <div className={styles.form_group} style={{ flex: 1 }}>
              <label>Year</label>
              <input type="number" className={styles.input} value={year} onChange={e => setYear(e.target.value)} />
            </div>
            <div className={styles.form_group} style={{ flex: 1 }}>
              <label>Release Date (YYYY-MM-DD)</label>
              <input type="text" className={styles.input} value={releaseDate} onChange={e => setReleaseDate(e.target.value)} placeholder="1975-06-20" />
            </div>
          </div>

          <div style={{ display: 'flex', gap: '15px' }}>
            <div className={styles.form_group} style={{ flex: 1 }}>
              <label>Runtime (min)</label>
              <input type="number" className={styles.input} value={runtime} onChange={e => setRuntime(e.target.value)} />
            </div>
            <div className={styles.form_group} style={{ flex: 1 }}>
              <label>Budget ($)</label>
              <input type="number" className={styles.input} value={budget} onChange={e => setBudget(e.target.value)} />
            </div>
          </div>

          <div className={styles.form_group}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '5px' }}>
              <label style={{ marginBottom: 0 }}>Subtitles * (.srt, .txt)</label>
              
              <input 
                type="file" 
                accept=".srt,.txt" 
                ref={fileInputRef} 
                style={{ display: 'none' }} 
                onChange={handleFileUpload} 
              />
              <button 
                type="button" 
                onClick={() => fileInputRef.current?.click()}
                style={{ 
                  background: 'rgba(255,255,255,0.1)', border: 'none', color: '#fff', 
                  padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '11px' 
                }}
              >
                Upload File
              </button>
            </div>
            
            <textarea 
              required
              className={styles.textarea} 
              value={subtitles} 
              onChange={e => setSubtitles(e.target.value)}
              placeholder="Paste subtitle text manually or click 'Upload File'"
              style={{ height: '100px', fontSize: '12px', fontFamily: 'monospace' }}
            />
          </div>

          <div className={styles.form_group}>
            <label>Overview</label>
            <textarea 
              className={styles.textarea} 
              value={overview} 
              onChange={e => setOverview(e.target.value)}
              placeholder="Brief plot summary..."
              style={{ height: '60px' }}
            />
          </div>

          <button type="submit" className={styles.submit_btn} disabled={loading || !title.trim() || !subtitles.trim()}>
            {loading ? 'Submitting...' : 'Submit for Review'}
          </button>

          {error && <div className={styles.error_msg}>{error}</div>}
          {success && <div className={styles.success_msg}>Movie successfully added to queue!</div>}
        </form>
      </div>
    </div>
  );
}