// src/pages/HomePage.tsx
import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import styles from './HomePage.module.css';
import { PlotlyChart } from '../components/Charts/PlotlyChart';

// Данные дляストーリー-слайдов (Scrollytelling)
const STORY_SLIDES = [
  {
    id: 1,
    title: "The Explosion of Cinema: Setting the Stage",
    text: [
      "Over the past century, cinema has undergone a fundamental transformation — from a relatively small-scale industry into a massive global content engine. For decades, film production grew steadily, shaped by technological breakthroughs like sound and the rise of blockbuster storytelling. But the real inflection point comes in the modern era, where streaming platforms dramatically accelerated output, pushing the number of films to unprecedented levels.",
      "This shift matters because scale changes everything. When thousands of stories compete for attention each year, success is no longer just about telling a good story — it’s about telling one that stands out emotionally. To understand how cinema adapted to this pressure, we need to look inside the stories themselves, starting with the language they use."
    ],
    apiEndpoint: "http://localhost:5555/api/charts/movies-per-year"
  },
  {
    id: 2,
    title: "The Language of Cinema: Familiar Words, Strategic Impact",
    text: [
      "At first glance, movie dialogue appears deceptively simple. The most common words across films are highly conversational — words like “hey,” “love,” and “sorry” dominate, reflecting how cinema mirrors everyday human interaction. This creates accessibility and realism, allowing audiences to immediately connect with characters and situations.",
      "However, when we shift from frequency to importance, a different picture emerges. The words that truly define films are not the most common ones, but those tied to emotional intensity, conflict, and relationships. Profanity, family roles, and authority figures rise to the top, signaling that what makes dialogue impactful is not how often words are used, but how much emotional weight they carry.",
      "This reveals a key principle: cinema uses simple language as a foundation, but relies on emotionally charged moments to create impact. That naturally leads to a deeper question — how have these emotional patterns evolved over time?"
    ],
    apiEndpoint: "http://localhost:5555/api/charts/vocabulary-tfidf"
  },
  {
    id: 3,
    title: "Emotional Volatility: The Rise of Intensity",
    text: [
      "As cinema scaled, it didn’t just produce more content — it produced more emotionally dynamic content. Over time, we observe a clear upward trend in emotional volatility, meaning that stories increasingly move through stronger and more varied emotional states. This isn’t random; the spikes often align with periods of societal change or cinematic reinvention, such as wartime eras or the rise of more experimental filmmaking in the late 20th century.",
      "What this suggests is that audiences have gradually developed a higher tolerance — and even an expectation — for emotional intensity. Stories are no longer static or linear; they are designed to take viewers on a more turbulent journey.",
      "But intensity alone doesn’t define storytelling. The critical question is how these emotions are balanced — and whether there is a consistent emotional direction underlying this complexity."
    ],
    apiEndpoint: "http://localhost:5555/api/charts/macro-seismograph"
  },
  {
    id: 4,
    title: "The Emotional Balance: Why Stories Still Feel Good",
    text: [
      "Despite the increasing complexity and intensity of emotions, one pattern remains remarkably consistent: cinema overwhelmingly leans toward positive sentiment. Across decades, positive emotional expression consistently outweighs negative, creating a stable imbalance that persists regardless of era or genre trends.",
      "This highlights a fundamental truth about storytelling. Audiences may be drawn in by conflict, tension, and struggle, but they ultimately seek resolution. The emotional journey can be chaotic, but the destination is often reassuring. In other words, films are not just about making audiences feel — they are about making them feel better by the end.",
      "To understand how this plays out within individual stories, we need to move from overall sentiment to the actual flow of emotions within a narrative."
    ],
    apiEndpoint: "http://localhost:5555/api/charts/sentiment-ring"
  },
  {
    id: 5,
    title: "The Storytelling Engine: From Conflict to Resolution",
    text: [
      "When we examine how emotions evolve within films, a clear structure emerges. Stories may begin in a wide range of emotional states — from joy to fear to anger — but they overwhelmingly converge toward a positive ending. This is not a subtle tendency; it is a dominant pattern. Regardless of where a story starts, the probability that it ends in joy is significantly higher than any other outcome.",
      "What’s more, this transformation is not abrupt but unfolds across a structured progression. Emotional flows show that stories typically pass through a phase of tension or instability before resolving, reinforcing the idea that conflict is a necessary step toward satisfaction.",
      "This reveals the core engine of cinematic storytelling: emotional contrast drives engagement, but resolution delivers payoff. And if this structure is so consistent, the next logical question is whether it actually translates into real-world success."
    ],
    apiEndpoint: "http://localhost:5555/api/charts/emotion-transition"
  },
  {
    id: 6,
    title: "Emotion Strategy: Safe Bets vs High-Risk Hits",
    text: [
      "When we connect emotional patterns to revenue, a more nuanced picture emerges — one that goes beyond simple “happy stories perform better.”",
      "Films dominated by joy tend to deliver consistent and stable outcomes. They rarely produce extreme highs, but they also avoid major failures. In many ways, joy acts as a “safe strategy” — it aligns with audience expectations and reliably generates moderate success.",
      "However, the data also reveals something more interesting. Some of the highest-grossing films are not purely joyful — they are emotionally heavier, often dominated by sadness or strong contrast between emotions. These films can achieve exceptional performance, but they are far less predictable. For every breakout success, there are many that fail to resonate.",
      "This creates a clear strategic trade-off. Studios can optimize for consistency, using emotionally positive narratives that reliably perform, or they can take on emotional risk, crafting deeper, more complex stories that have the potential to significantly outperform — but with much higher uncertainty. In other words, emotion is not just a storytelling tool — it is a portfolio strategy."
    ],
    apiEndpoint: "http://localhost:5555/api/charts/revenue-topology"
  },
  {
    id: 7,
    title: "Beyond One Shape: Toward a Data-Driven Story Framework",
    text: [
      "Kurt Vonnegut famously suggested that all stories can be reduced to a small number of universal shapes. And at a high level, our analysis supports this idea — emotional arcs do tend to cluster into recognizable patterns.",
      "But when we look closer, the reality is more nuanced. Instead of a handful of rigid story types, we observe multiple overlapping clusters, each representing a different variation of how stories evolve emotionally. These clusters are not isolated — they blend, intersect, and adapt depending on genre, era, and creative choices.",
      "This means that storytelling is not about choosing a single “correct” shape. It’s about navigating a landscape of possible emotional trajectories, each with different implications for audience engagement and success.",
      "Rather than forcing stories into predefined categories, we model them as dynamic paths in an emotional space — allowing us to compare, cluster, and understand narratives at scale. This forms the foundation of our core product: a graph-based representation of storytelling."
    ],
    apiEndpoint: "http://localhost:5555/api/charts/vonnegut-map"
  }
];

export function HomePage() {
  const observerRef = useRef<IntersectionObserver | null>(null);

  useEffect(() => {
    // Настройка Intersection Observer для анимаций появления при скролле
    observerRef.current = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add(styles.visible);
          entry.target.classList.remove(styles.hidden);
          observerRef.current?.unobserve(entry.target);
          
          setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
          }, 300);
        }
      });
    }, {
      threshold: 0.5,
      rootMargin: "0px 0px -50px 0px"
    });

    const hiddenElements = document.querySelectorAll(`.${styles.hidden}`);
    hiddenElements.forEach(el => observerRef.current?.observe(el));

    return () => observerRef.current?.disconnect();
  }, []);

  return (
    <div className={styles.page_wrapper}>
      
      {/* 1. HERO SECTION */}
      <section className={styles.hero}>
        <h1 className={styles.title}>CineGraph</h1>
        <p className={styles.subtitle}>
          A data-driven web application that processes raw subtitles from 40,000 movies 
          to generate interactive "Emotional Seismographs".
        </p>
        
        <Link to="/graph" className={styles.cta_button}>
          Launch Graph Explorer
        </Link>

        <div className={styles.scroll_indicator}>
          <span>Discover The Story</span>
          <div style={{ marginTop: '5px' }}>↓</div>
        </div>
      </section>

      {/* 2. ABOUT PROJECT (Based on README) */}
      <section className={`${styles.about_section} ${styles.hidden}`}>
        <h2>About The Project</h2>
        <p className={styles.about_text}>
          <b>CineGraph</b> transforms raw cinematic data into emotional insights. By processing tens of thousands of movie scripts, we've built a system that combines a massive scraping pipeline with NLP analysis and advanced clustering techniques to reveal the hidden structure of storytelling.
        </p>
        <p className={styles.about_text}>
          Inspect emotionally close movies through our interactive graph and dive deep into individual "sentiment arcs" consisting of 6 main emotions evolving over time.
        </p>
        
        <div className={styles.tech_stack}>
          <span className={styles.tech_badge} style={{color: '#3670A0', borderColor: '#3670A0'}}>Python</span>
          <span className={styles.tech_badge} style={{color: '#00D4AA', borderColor: '#00D4AA'}}>LangChain</span>
          <span className={styles.tech_badge} style={{color: '#4169e1', borderColor: '#4169e1'}}>PostgreSQL</span>
          <span className={styles.tech_badge} style={{color: '#43B02A', borderColor: '#43B02A'}}>Selenium</span>
          <span className={styles.tech_badge} style={{color: '#0db7ed', borderColor: '#0db7ed'}}>Docker</span>
        </div>
      </section>

      {/* 3. SCROLLYTELLING CONTENT */}
      <div style={{ paddingBottom: '100px', overflowX: 'hidden' }}>
        {STORY_SLIDES.map((slide) => (
          <section key={slide.id} className={`${styles.slide_section} ${styles.hidden}`}>
            
            <div className={styles.slide_text_content}>
              <h2 className={styles.slide_title}>
                {/* Опционально: можно сделать номер полупрозрачным для стиля */}
                <span style={{ color: 'rgba(255, 255, 255, 0.2)', marginRight: '15px', fontWeight: 900 }}>
                  0{slide.id}.
                </span> 
                {slide.title}
              </h2>
              {slide.text.map((paragraph, i) => (
                <p key={i} className={styles.slide_paragraph}>{paragraph}</p>
              ))}
            </div>

            {/* Контейнер для интерактивного Plotly-графика */}
            <div className={styles.chart_container}>
              <PlotlyChart apiEndpoint={slide.apiEndpoint} />
            </div>

          </section>
        ))}
      </div>
      
      {/* FINAL CTA & FOOTER */}
      <section className={`${styles.section} ${styles.hidden}`} style={{ textAlign: 'center', padding: '100px 20px' }}>
        <h2 style={{ border: 'none' }}>Ready to explore the graph?</h2>
        <Link to="/graph" className={styles.cta_button} style={{ marginTop: '20px', display: 'inline-block' }}>
          Open Graph Explorer
        </Link>
      </section>

      <footer style={{ textAlign: 'center', padding: '60px 40px', color: '#555', borderTop: '1px solid #222' }}>
        <p>© 2024 CineGraph Team. Data-Wrangling-and-Visualization.</p>
        <p style={{ fontSize: '0.9rem', marginTop: '10px' }}>
          <a href="https://github.com/Data-Wrangling-and-Visualization-2026/CineGraph" target="_blank" rel="noreferrer" style={{ color: '#888', textDecoration: 'none' }}>
            View Documentation on GitHub
          </a>
        </p>
      </footer>

    </div>
  );
}