"use client";

import {
  ArrowUp,
  Bot,
  Building2,
  Clock3,
  Database,
  FileText,
  LineChart,
  Search,
  User
} from "lucide-react";
import { FormEvent, useMemo, useRef, useState } from "react";

type Source = {
  title: string;
  url: string;
  ticker?: string;
  sourceType: "filing" | "news" | "price";
  score?: number;
  excerpt?: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

const starterQuestions = [
  "What risks did Apple disclose about supply chain and manufacturing?",
  "What changed in Apple's latest 10-Q?",
  "Summarize recent risks for AAPL from filings.",
  "What context should I know before looking at today's AAPL move?"
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Ask about a company, a filing, or a market move. I will retrieve local pgvector context and answer with citations."
    }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const latestSources = useMemo(() => {
    return [...messages].reverse().find((message) => message.sources?.length)?.sources ?? [];
  }, [messages]);

  async function submitQuestion(question: string) {
    const trimmed = question.trim();
    if (!trimmed || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed
    };
    setMessages((current) => [...current, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed })
      });
      if (!response.ok) {
        throw new Error(`Chat request failed: ${response.status}`);
      }
      const data = (await response.json()) as { answer: string; sources?: Source[] };
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.answer,
          sources: data.sources ?? []
        }
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            error instanceof Error
              ? `I could not reach the chat backend yet. ${error.message}`
              : "I could not reach the chat backend yet."
        }
      ]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitQuestion(input);
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <LineChart size={20} />
          </div>
          <div>
            <h1>Finance AI</h1>
            <p>SEC filings, prices, news</p>
          </div>
        </div>

        <section className="panel">
          <div className="panel-title">
            <Database size={16} />
            <span>Data Status</span>
          </div>
          <div className="status-row">
            <span>pgvector</span>
            <strong>Connected soon</strong>
          </div>
          <div className="status-row">
            <span>Embeddings</span>
            <strong>Ollama</strong>
          </div>
          <div className="status-row">
            <span>Universe</span>
            <strong>AAPL test</strong>
          </div>
        </section>

        <section className="panel sources-panel">
          <div className="panel-title">
            <FileText size={16} />
            <span>Latest Sources</span>
          </div>
          {latestSources.length === 0 ? (
            <p className="muted">Citations from retrieved filings will appear here.</p>
          ) : (
            <div className="source-list">
              {latestSources.map((source, index) => (
                <a key={`${source.url}-${index}`} href={source.url} target="_blank" rel="noreferrer" className="source-item">
                  <span className="source-index">[{index + 1}]</span>
                  <span>
                    <strong>{source.title}</strong>
                    <small>
                      {source.ticker ?? source.sourceType}
                      {typeof source.score === "number" ? ` · ${source.score.toFixed(2)}` : ""}
                    </small>
                  </span>
                </a>
              ))}
            </div>
          )}
        </section>
      </aside>

      <section className="chat-workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Local RAG Workspace</p>
            <h2>Ask grounded finance questions</h2>
          </div>
          <div className="topbar-meta">
            <span>
              <Building2 size={15} />
              SEC
            </span>
            <span>
              <Clock3 size={15} />
              Daily bars
            </span>
          </div>
        </header>

        <div className="messages">
          {messages.map((message) => (
            <article key={message.id} className={`message ${message.role}`}>
              <div className="avatar" aria-hidden="true">
                {message.role === "assistant" ? <Bot size={18} /> : <User size={18} />}
              </div>
              <div className="message-body">
                <p>{message.content}</p>
                {message.sources && message.sources.length > 0 ? (
                  <div className="citation-strip">
                    {message.sources.slice(0, 4).map((source, index) => (
                      <a key={`${source.url}-${index}`} href={source.url} target="_blank" rel="noreferrer">
                        [{index + 1}] {source.ticker ?? source.sourceType}
                      </a>
                    ))}
                  </div>
                ) : null}
              </div>
            </article>
          ))}
          {isLoading ? (
            <article className="message assistant">
              <div className="avatar" aria-hidden="true">
                <Bot size={18} />
              </div>
              <div className="message-body loading-text">Retrieving filings and drafting answer...</div>
            </article>
          ) : null}
        </div>

        <div className="starter-row">
          {starterQuestions.map((question) => (
            <button key={question} type="button" onClick={() => void submitQuestion(question)}>
              <Search size={15} />
              <span>{question}</span>
            </button>
          ))}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask about filings, risks, prices, or market moves..."
            rows={2}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void submitQuestion(input);
              }
            }}
          />
          <button type="submit" disabled={!input.trim() || isLoading} aria-label="Send message">
            <ArrowUp size={20} />
          </button>
        </form>
      </section>
    </main>
  );
}
