@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@		    Modify the following variables.                  @@@
@@@	        See the file manual.txt for documentation.           @@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@ Modify as needed @
new , 20000;
screen on;

@ Put the name of the output file after the = sign. @
output file=c:\cars\jae\tab2out.txt reset;

print " ";
print "Size is redefined.";
print "Size, luggsp, non-EV dummy, and non-CNG dummy";
print "are normal. Other coefficients fixed.";
print "250 reps. Seed=43.";
print "Logit starting values plus reasonable values for std devs.";

@ Number of observations. (integer) @
NOBS = 4654;

@ Maximium number of alternatives. (integer) @
NALT = 6;

@ Create your data matrix XMAT here.  It must contain all of the @
@ necessary variables, including the dependent variable and censoring @
@ variable (if needed.) @

@ File containing data. @
load XMAT[NOBS,(26*6)] = c:\cars\jae\xmat.txt;

@----------------------------------------------------------------@

@ Number of variables in XMAT. (integer) @
NVAR = 26;

@ Specify the variable in XMAT that is the dependent variable. (integer) @
IDDEP = 23;

@ Give the number of explanatory variables that have fixed coefficients. @
@ (integer) @
NFC = 19;

@ NFCx1 vector to identify the variables in XMAT that have fixed @
@ coefficients. (column vector) @
IDFC = { 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
               20, 21 };

@ Give the number of explanatory variables that have normally @
@ distributed coefficients. (integer) @
NNC = 4;

@ NNCx1 vector to identify the variables in XMAT that have normally @
@ distributed coefficients. (column vector) @
IDNC = { 6, 8, 24, 25 };

@ Give the number of explanatory variables that have log-normally @
@ distributed coefficients. (integer) @
NLC = 0;

@ NLCx1 vector to identify the variables in XMAT that have @
@ log-normally distributed coefficients. (column vector) @
IDLC = { 0 };

@ 1 if all people do not face all NALT alternatives, else 0. @
CENSOR = 0;

@ Identify the variable in XMAT that identifies the alternatives @
@ each person faces. (integer) @
IDCENSOR = { 0 }; 

@ 1 to print out the diagonal elements of the Hessian, else 0. @
PRTHESS = 1;

@ 1 to rescale any variable in XMAT, else 0. @
RESCALE = 1;

@ If RESCALE = 1 create a q x 2 matrix to id which variables to @
@ rescale and by what multiplicative factor, else make RESCLMAT @
@ a dummy value. @

RSCLMAT = { 2 .01, 
            3  .1, 
            4 .01,                     
            6 .1,
            9 .1 }; 



@ (NFC+(2*NNC)+(2*NLC)) x 1 vector of starting values. @
B = {-.185, .35, -.716, .26, -.445, .143, -.77, .413,
         .820, .637, -1.44, -1.02, -.799, -.179,
        .198, .443, .345, .313, .228,
      .93, .3, .50, .2, 
         0, 0.5, 0, -0.5
          } ; 
 
@ 1 to constrain any parameter to its starting value, else 0. @
CZVAR = 1;

@ If CZVAR = 1, create a vector of zeros and ones of the same @
@ dimension as B identifying which parameters are to vary, else make @
@ BACTIVE = { 0 }. (column vector) @
BACTIVE = ones(27,1);
BACTIVE[26,1]=0;
BACTIVE[24,1]=0;


@ 1 to constrain any of the error components to equal each other, @
@ else 0. (integer) @
CEVAR = 0;

@ If CEVAR=1, create a matrix of ones and zeros to constrain the @
@ necessary random parameters to be equal, else make EQMAT=0. (matrix)@
EQMAT = { 0 }; 

@ Number of repetitions.  NREP must be >= 1. (integer) @
NREP = 250;

@ Seed to use in random number generator. SEED1 must be >= 0. (integer) @
SEED1 = 43;

@ Maximum number of iterations in the maximization. (integer) @
NITER = 120;

@ Tolerance for convergence. (small decimal) @
EPS = 1.e-4;

@ If you want to forecast instead of estimating, set FCAST to 1, @
@ else 0. @
FCAST = 0;

@ If FCAST=1 and you want to weight the sample, set DOWGT to one, @
@ else 0. @
DOWGT = 0;

@ If DOWGT=1, specify a NOBSx1 vector of weights. (column vector) @
WGT = { 0 };

@ 1 to check the above inputs for conformability and consistency, @
@ else 0. @
VERBOSE = 1;

@ Identify the optimization routine: 1 Paul Ruud's Newton-Raph @
@                                    2 for Gauss's maxlik      @
OPTIM = 1; 

@ Specify the optimization algorithm/hessian calculation.(integer) @
METHOD = 1;

@ 1 if OPTIM=1 and want to use modified BHHH, else 0. @
MBHHH = 1;


@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@	You should not need to change anything below this line	     @@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@ Check inputs if VERBOSE=1 @
if ((VERBOSE /= 1) and (VERBOSE /= 0));
  print "VERBOSE must 0 or 1.";
  print "Program terminated.";
   stop;
endif;
if VERBOSE;
  pcheck;
else;
 print "The inputs have not been checked since VERBOSE=0.";
 print;
endif; 

@ Rescale the variables. @
if RESCALE;
  j = rows(RSCLMAT);
  i = 1;
  if VERBOSE;
    if (RSCLMAT[.,1] > NVAR);
      print "RSCLMAT identifies a variable that is not in the data set.";
      print "Program terminated.";
      stop;
    endif;
    print "Rescaling Data:";
    print "        Variable      Mult. Factor";
    do while i <= j;
      RSCLMAT[i,1] RSCLMAT[i,2];
      XMAT[.,(RSCLMAT[i,1]-1)*NALT+1:RSCLMAT[i,1]*NALT] = 
      XMAT[.,(RSCLMAT[i,1]-1)*NALT+1:RSCLMAT[i,1]*NALT] * RSCLMAT[i,2];
      i = i + 1;
    endo;
    print;
  else;
    do while i <= j;
      XMAT[.,(RSCLMAT[i,1]-1)*NALT+1:RSCLMAT[i,1]*NALT] =
      XMAT[.,(RSCLMAT[i,1]-1)*NALT+1:RSCLMAT[i,1]*NALT] * RSCLMAT[i,2];
      i = i + 1;
    endo;
  endif;
endif;

@ Print out starting values if VERBOSE = 1. @
if VERBOSE;
  print "The model has" NFC " fixed coefficients, for variables" IDFC;
  print "             " NNC "normally distributed coefficients, for variables" IDNC;
  print "          and" NLC "log-normally distributed coefficients, for variables" IDLC;
  print;
  if FCAST;
    print "Parameter values used in forecasting are:";
    print B;
  endif;
  if CZVAR and (FCAST == 0);
    print "Starting values:" ;
    print B;
    print;
    print "Parameters that are estimated (1) or held at starting value (0):";
    print BACTIVE;
    print;
  endif;
  if (CZVAR == 0) and (FCAST == 0);
    print "Starting values:";
    print B;
    print;
    print "All parameters are estimated; none is held at its starting value.";
    print;
  endif;
endif;

@ Create new B and BACTIVE with reduced lengths if @
@ user specifies equality constraints. @
if CEVAR;
  if CZVAR;
   BACTIVE = EQMAT * BACTIVE;
   BACTIVE = BACTIVE .> 0;
  endif;
  B = (EQMAT * B) ./ (EQMAT * ones((NFC+2*NNC+2*NLC),1));
  if VERBOSE and CZVAR and (FCAST == 0);
   print;
   print "Some parameters are constrained to be the same.";
   print "Starting values of constrained parameters:";
   print B;
   print;
   print "Constrained parameters that are estimated (1) or held at starting value (0):";
   print BACTIVE;
   print;
  endif;
  if VERBOSE and (CZVAR == 0) and (FCAST == 0);
   print "Some parameters are constrained to be the same.";
   print "Starting values of constrained parameters:";
   print B;
   print;
   print "All constrained parameters are estimated; none is held at its starting value.";
   print;
  endif;
endif;

@ Describing random terms. @
if VERBOSE;
  print "Random error terms are based on:";
  print "Seed:  " SEED1;
  print "Repetitions:  " NREP;
  print;
  print "Number of random draws for each observation in each repetition:" (NNC+NLC);
  print;
endif;
NECOL = maxc( ones(1,1) | (NNC+NLC) );

@ Do forecasting @
if FCAST;
  beta = forecst(B,XMAT);
  print "Forecasted shares for each alternative:";
  print beta;
  stop;
endif;

@ Set up and do maximization @
if OPTIM == 1;
  beta = domax(&ll,&gr,B,BACTIVE); 
  print;
  print "Remember: Signs of standard deviations are irrelevant.";
  print "Interpret them as being positive.";
  print;
endif;

if OPTIM == 2;
  library maxlik,pgraph;
  #include maxlik.ext;
  maxset;
  _max_GradTol = EPS;
  _max_GradProc = &gr; 
  _max_MaxIters = NITER;
  _max_Algorithm = METHOD;
  if CZVAR;
     _max_Active = BACTIVE;
  endif;
  {beta,f,g,cov,ret} = maxlik(XMAT,0,&ll,B);
   call maxprt(beta,f,g,cov,ret);
  print;
  print "Remember: Signs of standard deviations are irrelevant.";
  print "Interpret them as being positive.";
  print;
  if (CZVAR == 0);
      print "gradient(hessian-inverse)gradient is:" ((g/_max_FinalHess)'g);
    else;
      g = selif(g,BACTIVE);
      print "gradient(hessian-inverse)gradient is:" ((g/_max_FinalHess)'g);
  endif;
  if PRTHESS;
    print "diagonal of hessian:" ((diag(_max_FinalHess))');
    print;
  endif; 
endif; 

@ THIS IS THE END OF THE MAIN PROGRAM. @
@ PROCS ll, gr, gpnr, fcast, expand, domax, pcheck follow.@
/* LOG-LIKELIHOOD FUNCTION */
proc ll(b,x);

  @ Relies on the globals: CENSOR, CEVAR, EQMAT, IDCENSOR, IDDEP,  @
  @      IDFC, IDLC, IDNC, NALT, NECOL, NFC, NLC, NNC, NOBS, NREP, @
  @      SEED1, XMAT                                               @

  local c, k, r, y, km, bn;
  local v, ev, p0, err, seed2; 

  if CEVAR;
    bn = EQMAT' * b;            @ Expand b to its original size. @
  else;
    bn = b;
  endif;

  v = zeros(NOBS,NALT);		@ Argument to logit formula @
 
  p0 = zeros(NOBS,1);		@ Scaled simulated probability	@
  y = (IDDEP-1)*NALT;           @ Identifies dependent variable @
  if CENSOR; 
    c = (IDCENSOR-1)*NALT;      @ Identifies censor variable @
  endif;

  seed2=SEED1;

  k = 1;
  do while k <= NFC;           @ Adds variables with fixed coefficients @
    km = (IDFC[k,1]-1)*NALT;
    v = v + bn[k] .* XMAT[.,(km+1):(km+NALT)];
    k = k+1;
  endo;

  r = 1;
  do while r <= NREP;          @ Repetitions for random coefficients. @
    ev = v;                   
    err = rndns(NOBS,NECOL,seed2);

    k = 1;
    do while k <= NNC;         @ Adds variables with normal coefficients @
      km = (IDNC[k,1]-1)*NALT;
      ev = ev + (bn[NFC+(2*k)-1] + (bn[NFC+(2*k)] .* err[.,k]))
                   .* XMAT[.,(km+1):(km+NALT)];
      k = k+1;
    endo;

    k = 1;
    do while k <= NLC;         @ Adds variables with log-normal coefficients @
      km = (IDLC[k,1]-1)*NALT;
      ev = ev + exp(bn[NFC+(2*NNC)+(2*k)-1]
                      +(bn[NFC+(2*NNC)+(2*k)] .* err[.,(NNC+k)]))
                      .* XMAT[.,(km+1):(km+NALT)];
      k = k+1;
    endo;

    ev = exp(ev);

    if CENSOR;
         p0 = p0 + (sumc((ev .* XMAT[.,(y+1):(y+NALT)])')) 
                ./ (sumc((ev .* XMAT[.,(c+1):(c+NALT)])'));
    else; 
         p0 = p0 + (sumc((ev .* XMAT[.,(y+1):(y+NALT)])')) 
                ./ (sumc(ev'));
    endif;
    r = r+1;
  endo;

  retp(ln(p0./NREP));
endp;

/* GRADIENT */
proc gr(b,x);

  @ Relies on the globals: CENSOR, CEVAR, EQMAT, IDCENSOR, IDDEP,  @
  @      IDFC, IDLC, IDNC, NALT, NECOL, NFC, NLC, NNC, NOBS, NREP, @
  @      SEED1, XMAT                                               @

  local i, j, k, r, km, l, part, bn;
  local v, ev, p1, p0, denom, der, err, seed2;

  if CEVAR;
    bn = EQMAT' * b;            @ Expand b to its original size. @
  else;
    bn = b;
  endif;

  v = zeros(NOBS,NALT);         @ Argument to logit formula    @

  p0 = zeros(NOBS,1);           @ Scaled simulated probability	@
  der = zeros(NOBS,NFC+(2*NNC)+(2*NLC)); @ Derivatives of probabilities	@
  i = (IDDEP-1)*NALT;           @ Identifies dependent variable @
  if CENSOR; 
   j = (IDCENSOR-1)*NALT;       @ Identifies censor variable @
  endif;

  seed2=SEED1;

  k = 1;
  do while k <= NFC;            @ Adds variables with fixed coefficients @
    km = (IDFC[k,1]-1)*NALT;
    v = v + bn[k] .* XMAT[.,(km+1):(km+NALT)];
    k = k+1;
  endo;

  r = 1;
  do while r <= NREP;           @ Repetitions for random coefficients @
    ev = v;
    err = rndns(NOBS,NECOL,seed2);

    k = 1;
    do while k <= NNC;          @ Adds variables with normal coefficients @
      km = (IDNC[k,1]-1)*NALT;
      ev = ev + (bn[NFC+(2*k)-1] + (bn[NFC+(2*k)] .* ERR[.,k]))
                  .* XMAT[.,(km+1):(km+NALT)];
      k = k+1;
    endo;

    k = 1;
    do while k <= NLC;          @ Adds variables with log-normal coefficients @
      km = (IDLC[k,1]-1)*NALT;
      ev = ev + exp(bn[NFC+(2*NNC)+(2*k)-1]
                  +(bn[NFC+(2*NNC)+(2*k)] .* ERR[.,(NNC+k)]))
                  .* XMAT[.,(km+1):(km+NALT)];
      k = k+1;
    endo;

    ev = exp(ev);

    if CENSOR;
      denom = sumc( (ev .* XMAT[.,(j+1):(j+NALT)])' );
    else;
      denom = sumc( ev' );
    endif;
    p1 = sumc((ev .* XMAT[.,(i+1):(i+NALT)])') ./ denom;
    p0 = p0 + p1;

    if CENSOR;
      ev = (ev ./ denom) .* XMAT[.,(j+1):(j+NALT)];
    else;
      ev = (ev ./ denom);
    endif;

    @ Calculate grad for first NFC parameters @

    k = 1;
    do while k <= NFC;
      km = (IDFC[k,1]-1)*NALT;     
      part = (XMAT[.,(i+1):(i+NALT)] - ev) 
                .* XMAT[.,(km+1):(km+NALT)];
      der[.,k] = der[.,k] + (sumc(part') .* p1);
      k = k+1;
    endo;
   
    @ Calculate grad for next 2*NNC parameters @

    k = 1;
    do while k <= NNC;
      km = (IDNC[k,1]-1)*NALT;
      part = (XMAT[.,(i+1):(i+NALT)] - ev) 
                 .* XMAT[.,(km+1):(km+NALT)];
      der[.,NFC+(2*k)-1] = der[.,NFC+(2*k)-1] + (sumc(part') .* p1);

      part = (XMAT[.,(i+1):(i+NALT)] - ev) 
                 .* (ERR[.,k] .* XMAT[.,(km+1):(km+NALT)]);
      der[.,NFC+(2*k)] = der[.,NFC+(2*k)] + (sumc(part') .* p1);
      k = k + 1;
    endo;

    @ Calculate grad for next 2*NLC parameters @

    k = 1;
    do while k <= NLC;
      km = (IDLC[k,1]-1)*NALT;
      part = (XMAT[.,(i+1):(i+NALT)] - ev) 
       .* exp(bn[NFC+(2*NNC)+(2*k)-1]+(bn[NFC+(2*NNC)+(2*k)] .* ERR[.,(NNC+k)]))
                  .* XMAT[.,(km+1):(km+NALT)];
      der[.,NFC+(2*NNC)+(2*k)-1] = der[.,NFC+(2*NNC)+(2*k)-1] 
                 + (sumc(part') .* p1);

      part = (XMAT[.,(i+1):(i+NALT)] - ev)
       .* exp(bn[NFC+(2*NNC)+(2*k)-1]+(bn[NFC+(2*NNC)+(2*k)] .* ERR[.,(NNC+k)]))
                  .* (ERR[.,k] .* XMAT[.,(km+1):(km+NALT)]);
      der[.,NFC+(2*NNC)+(2*k)] = der[.,NFC+(2*NNC)+(2*k)] 
                 + (sumc(part') .* p1);
      k = k + 1;
    endo;
   
    r = r+1;
  endo;

  if CEVAR;
    retp((der ./ p0) * EQMAT' );
     else;
    retp(der ./ p0);     
  endif;

endp;

/* GRADIENT FOR PAUL RUUD'S ROUTINE WHEN USING NEWTON-RAPHSON*/
@ USE WHEN OPTIM == 1 AND METHOD == 2 @
proc gpnr(b);
  @ Relies on globals: XMAT  @
  local grad;
  grad = gr(b,XMAT);
  retp(sumc(grad));
endp;


/* FORECASTED SHARES */
proc forecst(b,x);
  @ Relies on the globals: DOWGT, CENSOR, CEVAR, EQMAT, IDCENSOR,  @
  @      IDFC, IDLC, IDNC, NALT, NECOL, NFC, NLC, NNC, NOBS, NREP, @
  @      SEED1, WGT, XMAT                                          @

  @ Relies on the globals: NOBS, NALT, NFC, NNC, NLC, IDFC, IDNC, IDLC, @
  @                         NREP, SEED1, FCAST, DOWGT, WGT, NECOL   	@

  local i, j, k, r, km, bn;
  local v, ev, p0, err, seed2; 

  if CEVAR;
    bn = EQMAT' * b;            @ Expand b to its original size. @
  else;
    bn = b;
  endif;

  v = zeros(NOBS,NALT);		@ Argument to logit formula @
 
  p0 = zeros(NOBS,NALT);	@ Scaled simulated probability	@
  if CENSOR; 
   j = (IDCENSOR-1)*NALT;       @ Identifies censor variable @
  endif;

  seed2=SEED1;

  k = 1;
  do while k <= NFC;           @ Adds variables with fixed coefficients @
    km = (IDFC[k,1]-1)*NALT;
    v = v + bn[k] .* XMAT[.,(km+1):(km+NALT)];
    k = k+1;
  endo;

  r = 1;
  do while r <= NREP;          @ Repetitions for random coefficients. @
    ev = v;                   
    err = rndns(NOBS,NECOL,seed2);

    k = 1;
    do while k <= NNC;         @ Adds variables with normal coefficients @
      km = (IDNC[k,1]-1)*NALT;
      ev = ev + (bn[NFC+(2*k)-1] + (bn[NFC+(2*k)] .* err[.,k]))
                   .* XMAT[.,(km+1):(km+NALT)];
      k = k+1;
    endo;

    k = 1;
    do while k <= NLC;         @ Adds variables with log-normal coefficients @
      km = (IDLC[k,1]-1)*NALT;
      ev = ev + exp(bn[NFC+(2*NNC)+(2*k)-1]
                      +(bn[NFC+(2*NNC)+(2*k)] .* err[.,(NNC+k)]))
                      .* XMAT[.,(km+1):(km+NALT)];
      k = k+1;
    endo;

    ev = exp(ev);

    if CENSOR;
         p0 = p0 + (ev .* XMAT[.,(j+1):(j+NALT)]) 
                ./ (sumc((ev .* XMAT[.,(j+1):(j+NALT)])'));
    else; 
         p0 = p0 + ev ./ (sumc(ev '));
    endif;
    r = r+1;
  endo;

  if DOWGT;
    retp(meanc(WGT .* (p0./NREP)));
  else;
    retp(meanc(p0./NREP));
  endif;
endp;

/* EXPANDS THE DIRECTION VECTOR; ALLOWS PARAMETERS TO STAY AT STARTING	*/
/* VALUES; HELPER PROCEDURE FOR &domax					*/
proc expand( x, e );
    local i,j;
    i = 0;
    j = 0;
    do while i < rows(e);
        i = i + 1;
        if e[i];
            j = j + 1;
            e[i] = x[j];
        endif;
    endo;
    if j/=rows(x); "Error in expand."; stop; endif;
    retp( e );
endp;


/* MAXIMIZATION ROUTINE COURTESY OF PAUL RUUD */
proc domax( &f, &g, b, bactive );

  @ Relies on the globals: CZVAR, EPS, METHOD, NITER, NOBS,  @
  @      PRTHESS, XMAT                                       @


  local f:proc, g:proc, first, grad, hesh, ihesh, direc, step, _tol, _biter;
  local _f0, _f1, _fold, lambda, nb, printer, repeat;

  _tol  = 1;
  _biter = 0;

  _f0    =  sumc(f( b, XMAT ));

  nb = seqa(1,1,rows(b));

  format /m1 /rdt 16,8;
  print; print; print;

  do while (_tol > EPS or _tol < 0) and (_biter < NITER);
    _biter = _biter + 1;

    print "==========================================================================";
    print "          Iteration: " _biter;
    print "          Function:  " _f0;

    if (METHOD == 1);
     grad = g( b, XMAT );
     hesh = grad'grad;
     grad = sumc(grad);
    endif;
 
    if (METHOD == 2);
     grad = gpnr( b );
     hesh  = -gradp( &gpnr, b );
    endif;
 

    @ Select only the variables that we want to maximize over @
    if CZVAR; 
      grad  = selif( grad, bactive );
      hesh  = selif( hesh, bactive );
      hesh  = selif( hesh', bactive );
    endif;

    if (det(hesh)==0);
      print "Singular Hessian!";
      print "Program terminated.";
      stop;
    else;
      ihesh = inv(hesh);
      direc = ihesh*grad;
    endif;

    _tol   = direc'grad;

    if CZVAR;
      direc = expand( direc, bactive);
    endif;

    print "          Tolerance: " _tol;
    print "--------------------------------------------------------------------------";
    if PRTHESS;
      if CZVAR;
        printer = nb~b~expand(grad./NOBS,bactive)~expand(diag(hesh),bactive);
      else;
        printer = nb~b~(grad./NOBS)~(diag(hesh));
      endif;

      print "                             Coefficients             Rel. Grad.               Hessian";

    else;
      if CZVAR;
        printer = nb~b~expand(grad./NOBS,bactive);
      else;
        printer = nb~b~(grad./NOBS);
      endif;

      print "                             Coefficients             Rel. Grad.";

    endif;
    print printer;

    if (_tol >= 0) and (_tol < EPS);        
      break;
    elseif _tol < 0;
      direc = -direc;
    endif;

    step = 2;
    lambda = .5;
    repeat = 1;
    first = 1;
    _f1 = _f0;

    steploop:				@ Start subroutine @
    step = step*lambda;
    _fold = _f1;
    _f1 = sumc(f( b+step*direc, XMAT ));
    print "--------------------------------------------------------------------------";
    print " Step: " step;
    print " Function: " _f1;
    if repeat;
      print " Change: " _f1-_f0;
    else;
      print " Change: " _f1-_fold;
    endif;
    if MBHHH;
      if (step < 1.e-5);
        print "Failed to find increase.";
        retp(b);
      elseif (_f1 <= _f0) and (repeat);
        first = 0;
        goto steploop;
      elseif (_f1 > _fold) and (first);
        lambda = 2;
        repeat = 0;
        goto steploop;
      endif;
    else;
      if (step < 1.e-5);
        print "Failed to find increase.";
        retp(b);
      elseif (_f1 <= _f0);
        goto steploop;
      endif;
    endif;

    if (repeat);
      b = b + step*direc;
      _f0 = _f1;
    else;
      b = b + .5*step*direc;
      _f0 = _fold;
    endif;

  endo;

  print "==========================================================================";
  print;

  format /m1 /rdt 1,8;
  if (_tol< EPS);
    print "Converged with tolerance:  " _tol;
    print "Funcation value:  " _f0;
  else;
    print "Stopped with tolerance:  " _tol;
    print "Function value:  " _f1;
  endif;

  print;
  lambda = eig(hesh);
  if lambda>0;
    print "Function is concave at stopping point.";
  else;
    print "WARNING:  Function is not concave at stopping point.";
  endif;
  print;

  if CZVAR;
    printer = nb~b~expand(sqrt(diag(ihesh)),bactive);
  else;
    printer = nb~b~sqrt((diag(ihesh)));
  endif;

  format /m1 /rdt 16,8;
  print "      Parameters               Estimates           Standard Errors";
  print "--------------------------------------------------------------------------";
  print printer;

  retp( b );
endp;


/* This proc checks the inputs if VERBOSE=1 */
proc (0) = pcheck;
  local i, j;

  @ Checking XMAT @
  if ( rows(XMAT) /= NOBS);
    print "XMAT has" rows(XMAT) "rows";
    print "but it should have NOBS="  NOBS   "rows.";
    print "Program terminated.";
    stop;
  elseif( cols(XMAT) /= (NVAR*NALT));
    print "XMAT has" cols(XMAT) "columns";
    print "but it should have NVARxNALT= " (NVAR*NALT) "columns.";
    print "Program terminated.";
    stop;
  else;
    print "XMAT has:";
    print "Rows:  " NOBS;
    print "Cols:  " NVAR*NALT ;
    print "Containing" NVAR "variables for " NALT " alternatives.";
    print;
  endif; 

  @ Checking dependent variable @
  if (FCAST == 0);
    if (IDDEP <= 0);
      print "The dependent variable cannot be located";
      print "since you did not set IDDEP to a strictly positive number.";
      print "Program terminated.";
      stop;
    endif;
    if (IDDEP > NVAR);
      print "The dependent variable cannot be located";
      print "since you set IDDEP larger than NVAR.";
      print "Program terminated.";
      stop;
    endif;
    i = (IDDEP-1)*NALT;
    if (sumc(XMAT[.,(i+1):(i+NALT)]') > 1);
      print "The dependent variable does not sum to one";
      print "over alternatives for each observation.";
      print "Program terminated.";
      stop;
    endif;
  endif;

  @ Check fixed coefficients @
  if (NFC /= 0);
    if (rows(IDFC) /= NFC);
      print "IDFC has" rows(IDFC) "rows when it should have NFC=" NFC "rows.";
      print "Program terminated.";
      stop;
    endif;
    if (cols(IDFC) /= 1);
      print "Commas are needed between the elements of IDFC.";
      print "Program terminated.";
      stop;
    endif;
    if (1-(IDFC <= NVAR));
      print "IDFC identifies a variable that is not in the data set.";
      print "All elements of IDFC should be <= NVAR, which is " NVAR;
      print "Program terminated.";
      stop;
    endif;
  endif;

  @ Check normally distributed coefficients @
  if (NNC /= 0);
    if (rows(IDNC) /= NNC);
      print "IDFC has" rows(IDNC) "rows when it should have NNC=" NNC "rows.";
      print "Program terminated.";
      stop;
    endif;
    if (cols(IDNC) /= 1);
      print "Commas are needed between the elements of IDNC.";
      print "Program terminated.";
      stop;
    endif;
    if (1-(IDNC <= NVAR));
      print "IDNC identifies a variable that is not in the data set.";
      print "All elements of IDNC should be <= NVAR, which is " NVAR;
      print "Program terminated.";
      stop;
    endif;
  endif;

  @ Check log-normally distributed coefficients @
  if (NLC /= 0);
    if (rows(IDLC) /= NLC);
      print "IDFC has" rows(IDLC) "rows when it should have NLC=" NLC "rows.";
      print "Program terminated.";
      stop;
    endif;
    if (cols(IDLC) /= 1);
      print "Commas are needed between the elements of IDLC.";
      print "Program terminated.";
      stop;
    endif;
    if (1-(IDLC <= NVAR));
      print "IDLC identifies a variable that is not in the data set.";
      print "All elements of IDLC should be <= NVAR, which is " NVAR;
      print "Program terminated.";
      stop;
    endif;
  endif;

  @ Checking censoring variable. @
  if ((CENSOR /= 1) and (CENSOR /= 0));
    print "CENSOR must be 0 or 1.";
    print "Program terminated.";
    stop;
  endif;
  if CENSOR;
    if IDCENSOR <= 0;
      print "The censoring variable cannot be located";
      print "since you did not set IDCENSOR to a strictly positive number.";
      print "Program terminated.";
      stop;
    endif;
    if IDCENSOR > NVAR;
      print "The censoring variable cannot be located";
      print "since you set IDCENSOR larger than NVAR.";
      print "Program terminated.";
      stop;
    endif;
    j = (IDCENSOR-1)*NALT;
    i = ((XMAT[.,(j+1):(j+NALT)] .== 1) .OR (XMAT[.,(j+1):(j+NALT)] .== 0) == 1);
    if (1-i);
      print "One or more elements of your censoring variable do not equal 0 or 1.";
      print "Program terminated.";
      stop; 
    endif;
    if (FCAST == 0);
      i = (IDDEP-1)*NALT;
      j = ((XMAT[.,(i+1):(i+NALT)] .AND (XMAT[.,(j+1):(j+NALT)] .== 0)) == 0);
      if (1-j);
        print "Your censoring variable eliminates the chosen alternative";
        print "for one or more observations.";
        print "Program terminated.";
        stop;
      endif;
    endif;
  endif; 

  @ Check RESCALE @
  if ((PRTHESS /= 1) and (PRTHESS /= 0));
    print "PRTHESS must be 0 or 1.";
    print "Program terminated.";
    stop;
  endif;
  if ((RESCALE /= 1) and (RESCALE /= 0));
    print "RESCALE must be 0 or 1.";
    print "Program terminated.";
    stop;
  endif;

  @ Check starting values @
  if (rows(B) /= (NFC+2*NNC+2*NLC));
    print "Starting values B has " rows(B) "rows";
    print "when it should have NFC+2*NNC+2*NLC= " (NFC+2*NNC+2*NLC) "rows.";
    print "Program terminated.";
    stop;
  endif;
  if (cols(B) /= 1);
    print "Commas needed between the elements of B.";
    print "Program terminated.";
    stop;
  endif;

  @ Check CZVAR @
  if ((CZVAR /= 1) and (CZVAR /= 0));
    print "CZVAR must be 0 or 1.";
    print "Program terminated.";
    stop;
  endif;
  if CZVAR;
    if (rows(BACTIVE) /= (NFC+2*NNC+2*NLC));
      print "BACTIVE has " rows(BACTIVE) "rows";
      print "when it should have NFC+2*NNC+2*NLC= " (NFC+2*NNC+2*NLC) "rows.";
      print "Program terminated.";
      stop;
    endif;
    if (cols(BACTIVE) /= 1);
      print "Commas needed between the elements of BACTIVE.";
      print "Program terminated.";
      stop;
    endif;
  endif;

  @ Check CEVAR @
  if ((CEVAR /= 1) and (CEVAR /= 0));
    print "CEVAR must be 0 or 1.";
    print "Program terminated.";
    stop;
  endif;
  if CEVAR;
    if (cols(EQMAT) /= (NFC+2*NNC+2*NLC));
      print "EQMAT has " cols(EQMAT) " columns";
      print "when it should have NFC+2*NNC+2*NLC=" (NFC+2*NNC+2*NLC) " columns.";
      print "Program terminated.";
      stop;
    endif;
    if (rows(EQMAT) >= (NFC+2*NNC+2*NLC));
      print "EQMAT has " rows(EQMAT) " rows";
      print "when it should have strictly less than NFC+2*NNC+2*NLC=" (NFC+2*NNC+2*NLC) " rows.";
      print "Program terminated.";
      stop;
    endif;
  endif;

 @ Checking NREP @
  if ( NREP <= 0 );
    print "Error in NREP:  must be positive.";
    print "Program terminated.";
    stop;
  endif;

  @ Check FCAST @
  if ((FCAST /= 1) and (FCAST /= 0));
    print "FCAST must be 0 or 1.";
    print "Program terminated.";
    stop;
  endif;
  if ((DOWGT /= 1) and (DOWGT /= 0));
    print "DOWGT must be 0 or 1.";
    print "Program terminated.";
    stop;
  endif;
  if (FCAST == 0) and (DOWGT == 1);
    print "Weights are only applied for forecasting.";
    print "You have DOWGT=1 when FCAST=0.";
    print "Program terminated.";
    stop;
  endif;
  if FCAST and DOWGT and (rows(WGT) /= NOBS);
    print "WGT has" rows(WGT) "rows but needs to have NOBS=" NOBS "rows.";
    print "Program terminated.";
    stop;
  endif;
  if FCAST and DOWGT and (cols(WGT) /= 1);
    print "WGT has" cols(WGT) "columns but needs to have 1 column.";
    print "Program terminated.";
    stop;
  endif;

  @ Check METHOD and OPTIM @
  if (FCAST == 0);
    if (METHOD < 1);
      print "METHOD must be 1-6.";
      print "Program terminated.";
      stop;
    endif;
    if (OPTIM /= 1) and (OPTIM /= 2);
      print "OPTIM must be 1 or 2.";
      print "Program terminated.";
      stop;
    endif;
    if ((OPTIM == 1) and (METHOD > 2));
      print "Method "  METHOD " is not an option with OPTIM = 1.";
      print "Program terminted.";
      stop;
    endif;
    if ((OPTIM == 2) and (METHOD > 6));
      print "Method " METHOD " is not an option with OPTIM = 2.";
      print "Program terminated.";
      stop;
    endif;
  endif;

  @ Check MBHHH @
  if ((MBHHH /= 1) and (MBHHH /= 0));
    print "MBHHH must be 0 or 1.";
    print "Program terminated.";
    stop;
  endif;

  print "The inputs pass all the checks and seem fine.";
  print;
  retp;
endp;
