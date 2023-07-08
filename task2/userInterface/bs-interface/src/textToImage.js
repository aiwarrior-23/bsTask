import React, { useState } from 'react';
import { Grid, Paper, Typography, TextField, Container, Button, Divider } from '@mui/material';
import axios from 'axios';


const TextToImage = () => {
    const [userInput, setUserInput] = useState('');
    const [response, setResponse] = useState('');
  
    const handleInputChange = (e) => {
      setUserInput(e.target.value);
    };

    const handleButtonClick = async () => {
        try {
          const response = await axios.post('http://localhost:5000/generate-image', {
            prompt: userInput,
            n: 2,
            size: "1024x1024",
          });
          setResponse(response.data.data[0].url);
        } catch (error) {
          console.error(error);
        }
      };

  return (
    <Grid container direction="column">
      <Grid item style={{ marginTop: "2%" }}>
        <Paper elevation={0}>
          <Typography variant='h4'>
            Please input your Prompt
          </Typography>
          <TextField label="Enter Text" variant="outlined" onChange={handleInputChange} style={{ marginTop: "2%", width: "50%" }} />
          <Container>
            <Button variant='contained' onClick={handleButtonClick} style={{ marginTop: "2%" }}>
              Submit
            </Button>
            <Divider style={{ margin: "2%" }} />
          </Container>
        </Paper>
      </Grid>
      <Grid item style={{ marginTop: "5%" }}>
        <Paper elevation={0}>
          <Typography variant='h4'>
            Here's your answer
          </Typography>
          <Typography style={{ marginTop: "2%" }}>
            {response && <img src={response} alt="Generated Image" />}
          </Typography>
        </Paper>
      </Grid>
    </Grid>
  );
};

export default TextToImage;
