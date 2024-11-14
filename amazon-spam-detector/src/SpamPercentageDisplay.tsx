import React from "react";
import {CircularProgressbar, buildStyles} from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

interface SpamPercentageDisplayProps {
    spamPercentage: number;
}

const SpamPercentageDisplay: React.FC<SpamPercentageDisplayProps> = ({spamPercentage}) => {
    return (
        <div className="spam-percentage">
            <h3>Spam Percentage</h3>
            <CircularProgressbar
                value={spamPercentage}
                text={`${spamPercentage.toFixed(1)}%`}
                styles={buildStyles({
                    pathColor: spamPercentage > 50 ? "red" : "green",
                    textColor: "#000",
                })}
            />
        </div>
    );
};

export default SpamPercentageDisplay;
