import { useMemo } from "react";
import PropTypes from "prop-types";
import "./SelectionArea.css";

const SelectionArea = ({
  className = "",
  selectionAreaBackgroundImage,
  rectangle,
  emptySelection,
}) => {
  const selectionAreaStyle = useMemo(() => {
    return {
      backgroundImage: selectionAreaBackgroundImage,
    };
  }, [selectionAreaBackgroundImage]);

  return (
    <div className={`selection-area ${className}`} style={selectionAreaStyle}>
      <img className="rectangle-icon17" alt="" src={rectangle} />
      <div className="select">
        <div className="selection-shapes" />
        <div className="empty-selection">{emptySelection}</div>
      </div>
    </div>
  );
};

SelectionArea.propTypes = {
  className: PropTypes.string,
  rectangle: PropTypes.string,
  emptySelection: PropTypes.string,

  /** Style props */
  selectionAreaBackgroundImage: PropTypes.string,
};

export default SelectionArea;
