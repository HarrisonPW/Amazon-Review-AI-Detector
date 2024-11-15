import {useState} from "react";
import axios from "axios";
import ReviewInput from "./components/ReviewInput";
import ResultsDisplay from "./components/ResultsDisplay";
import "./App.css";

interface Results {
    review: string;
    result_gpt2: string;
    result_nb: string;
    result_nb2: string;
    result_nb3: string;
    result_nb4: string;
    spam_percentage: number;
}

function App() {
    const [review, setReview] = useState<string>("");
    const [results, setResults] = useState<Results | null>(null);

    const handleSubmit = async () => {
        try {
            const response = await axios.post("http://localhost:5000/predict", {review});
            setResults(response.data);
        } catch (error) {
            console.error("Error fetching data:", error);
        }
    };

    return (
        <div className="app-container">
            <h1>Amazon Review Spam Detector</h1>
            <ReviewInput review={review} setReview={setReview} onSubmit={handleSubmit}/>
            {results && <ResultsDisplay results={results}/>}
        </div>
    );
}

export default App;
