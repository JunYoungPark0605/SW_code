import axios from 'axios';
import React, { useEffect, useState } from 'react';
import Axios from 'axios';
import Rate from 'rc-rate';
import 'rc-rate/assets/index.css';

function Star(props) {

    const [Stars, setStars] = useState(0)
    const [Save, setSave] = useState(0)
    let variable = {
        productId: props.productId,
        userId: props.userId,
        star: Star,
        save: Save,
    };

    const OnStar = (e) => {
      setStars(e)
    }    

    useEffect(() => {
      axios.post("api/star/getStar", variable).then((response) => {
        if (response.data.success) {
          console.log("response.data", response.data);

          if(response.data.stars.length > 0) {
            setSave(false);
            setStars(response.data.stars[0].star);
          }
          //별의 개수가 몇 개인지
        } else {
          alert("fail to getStars data")
        }
      })
    })



    useEffect(() => {
      axios.post("/api/star/upStar", variable).then((response) => {
        if (response.data.success) {
          setSave(false);
        } else {
          alert("fail to upStar")
        }
      })
    }, [Stars]);
  return (
    <div>
      <Rate onChange={OnStar} value={Stars} />
    </div>
  )
}

export default Star