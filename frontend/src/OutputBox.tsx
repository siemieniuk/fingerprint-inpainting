import React from "react";

function OutputBox(props: { image: string | null }) {
  return (
    <section>
      <div className="box-capture">Extracted fingerprint</div>
      <div className="box box-output" style={{
        backgroundImage: props.image ? `url(${props.image})` : 'none'
      }}>
        {props.image == null && (<>To be calculated</>)}
      </div>
    </section >
  )
}

export default OutputBox;