import {useState} from "react";
import axios from "axios";
import ReviewInput from "./components/ReviewInput";
import ResultsDisplay from "./components/ResultsDisplay";
import "./App.css";

interface Results {
    review: string;
    result_gpt2: string;
    result_lr: string;
    result_gemini: string;
    result_mnb: string;
    result_BERTLSTM: string;
    spam_percentage: number;
}

function App() {
    const [review, setReview] = useState<string>("");
    const [results, setResults] = useState<Results | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = async () => {
        try {
            setIsLoading(true);
            const response = await axios.post("http://localhost:88/predict", {review});
            setResults(response.data);
        } catch (error) {
            console.error("Error fetching data:", error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="app-container">
            <h1>Amazon Review Spam Detector</h1>
            <ReviewInput
                review={review}
                setReview={setReview}
                onSubmit={handleSubmit}
                disabled={isLoading}
            />
            {isLoading ? (
                <div className="flex flex-col items-center mt-8">
                    <div className="spinner"/>
                    <p className="text-gray-600 mt-2">Analyzing review...</p>
                </div>
            ) : results && <ResultsDisplay results={results}/>}
        </div>
    );
}

export default App;
