import React from "react";
import SpamPercentageDisplay from "../SpamPercentageDisplay.tsx";

interface ResultsDisplayProps {
    results: {
        result_gpt2: string;
        result_lr: string;
        result_gemini: string;
        result_mnb: string;
        result_nb4: string;
        spam_percentage: number;
    };
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({results}) => {
    const {result_gpt2, result_lr, result_gemini, result_mnb, result_nb4, spam_percentage} = results;

    return (
        <div className="results-display">
            <h2>Model Predictions</h2>
            <ul>
                <li>GPT-2 Model: {result_gpt2}</li>
                <li>LogisticRegression Model: {result_lr}</li>
                <li>Gemini-1.5 Model: {result_gemini}</li>
                <li>MultinomialNB Model: {result_mnb}</li>
                <li>MultinomialNB Model 4: {result_nb4}</li>
            </ul>
            <SpamPercentageDisplay spamPercentage={spam_percentage}/>
        </div>
    );
};

export default ResultsDisplay;
