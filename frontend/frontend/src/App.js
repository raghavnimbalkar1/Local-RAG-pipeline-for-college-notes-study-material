// src/App.js
import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]); // array of { role, content }
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef();

  const askQuestion = async () => {
    if (!question.trim()) return;
    const userMessage = { role: "user", content: question };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:8000/query", { question });
      const botMessage = { role: "assistant", content: res.data.answer };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      const botMessage = { role: "assistant", content: "Error: " + err.message };
      setMessages((prev) => [...prev, botMessage]);
    }

    setLoading(false);
  };

  // Scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  };

  return (
    <div className="App">
      <h1>StudyBuddy</h1>
      <div className="chat-window">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <div className="input-area">
        <textarea
          rows={2}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Type your question..."
          onKeyDown={handleKeyDown}
        />
        <button onClick={askQuestion} disabled={loading}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </div>
    </div>
  );
}

export default App;
