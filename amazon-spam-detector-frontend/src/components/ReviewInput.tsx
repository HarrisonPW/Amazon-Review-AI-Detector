import React from "react";

interface ReviewInputProps {
    review: string;
    setReview: (review: string) => void;
    onSubmit: () => void;
}

const ReviewInput: React.FC<ReviewInputProps> = ({review, setReview, onSubmit}) => {
    return (
        <div className="review-input">
      <textarea
          value={review}
          onChange={(e) => setReview(e.target.value)}
          placeholder="Enter an Amazon review to analyze..."
      />
            <button onClick={onSubmit}>Analyze Review</button>
        </div>
    );
};

export default ReviewInput;
