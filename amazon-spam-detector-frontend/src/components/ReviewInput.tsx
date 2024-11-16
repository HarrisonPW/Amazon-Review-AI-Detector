import React from "react";

interface ReviewInputProps {
    review: string;
    setReview: (review: string) => void;
    onSubmit: () => void;
    disabled: boolean;
}

const ReviewInput: React.FC<ReviewInputProps> = ({review, setReview, onSubmit, disabled}) => {
    return (
        <div className="review-input">
            <textarea
                value={review}
                onChange={(e) => setReview(e.target.value)}
                placeholder="Enter an Amazon review to analyze..."
                disabled={disabled}
                className={disabled ? "opacity-50" : ""}
            />
            <button
                onClick={onSubmit}
                disabled={disabled}
                className={disabled ? "opacity-50 cursor-not-allowed" : ""}
            >
                {disabled ? "Analyzing..." : "Analyze Review"}
            </button>
        </div>
    );
};

export default ReviewInput;
